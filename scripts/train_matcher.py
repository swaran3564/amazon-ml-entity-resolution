"""Training Baseline for Amazon ML Challenge 2026 Business Entity Resolution.

Architecture:
S1
 ├── Character TF-IDF retrieval (Top-20, char (3, 5))
 └── Word TF-IDF retrieval (Top-20, word (1, 2))
       ↓
 Candidate UNION (~28 candidates/S1)
       ↓
 ALL Pairwise Similarity Features (22 features via src/features.py)
       ↓
 LightGBM Classifier (with class_weight='balanced' & early stopping on PR-AUC)
       ↓
 Entity-Level Macro F0.5 Decision Threshold Optimization
       ↓
 Final Match Sets per S1
"""

import argparse
import gc
import heapq
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_union_blocking import (
    build_dual_consistent_vectorizers,
    get_peak_memory_mb,
    load_ground_truth_for_s1,
    stream_target_chunks,
)
from src.blocking import _extract_normalized_names
from src.config import (
    CHAR_TFIDF_TOP_K,
    LGBM_PARAMS,
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    WORD_NGRAM_RANGE,
    WORD_TFIDF_TOP_K,
)
from src.features import FEATURE_COLUMNS, build_feature_matrix


# =========================================================================
# 1. Entity-Level Status Classification & Labeling
# =========================================================================


def classify_s1_status(gt_set: Set[str], candidate_set: Set[str]) -> str:
    """Classify an S1 entity into one of three mutually exclusive statuses:

    - 'singleton': ground-truth match set is empty (zero-match entity)
    - 'matched_in_candidates': at least one GT match is present in the candidate set
    - 'matched_but_missed': GT is non-empty, but none of its true IDs were retrieved
    """
    if len(gt_set) == 0:
        return "singleton"
    elif len(gt_set & candidate_set) > 0:
        return "matched_in_candidates"
    else:
        return "matched_but_missed"


def assign_candidate_labels(
    pairs: Sequence[Tuple[str, str]],
    gt_map: Dict[str, Set[str]],
) -> List[int]:
    """Assign multi-match ground truth labels for candidate pairs.

    Every candidate whose entity_id appears in the S1 ground-truth set receives 1.
    All other candidates receive 0. Multiple positives per S1 are fully supported.
    """
    labels = []
    for sid, cid in pairs:
        is_pos = 1 if cid in gt_map.get(sid, set()) else 0
        labels.append(is_pos)
    return labels


# =========================================================================
# 2. Custom Entity-Level F0.5 Metric Evaluator
# =========================================================================


def compute_entity_f05(truth: Set[str], pred: Set[str], beta: float = 0.5) -> float:
    """Compute per-entity F0.5 score following exact Amazon ML Challenge rules:

    Cases:
       truth empty + pred empty -> F0.5 = 1.0
       truth empty + pred non-empty -> F0.5 = 0.0
       truth non-empty + pred empty -> F0.5 = 0.0
       otherwise:
           precision = TP / len(pred)
           recall = TP / len(truth)
           F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    """
    has_truth = len(truth) > 0
    has_pred = len(pred) > 0

    if not has_truth and not has_pred:
        return 1.0
    if not has_truth and has_pred:
        return 0.0
    if has_truth and not has_pred:
        return 0.0

    tp = len(truth & pred)
    if tp == 0:
        return 0.0

    precision = tp / len(pred)
    recall = tp / len(truth)

    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)


def evaluate_entity_macro_f05(
    s1_ids: Sequence[str],
    gt_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Compute macro-averaged F_beta across all specified S1 entities.

    Does NOT micro-average pairs; averages per-S1 F_beta scores uniformly.
    """
    per_s1_scores: Dict[str, float] = {}
    for sid in s1_ids:
        truth = gt_map.get(sid, set())
        pred = predictions_map.get(sid, set())
        per_s1_scores[sid] = compute_entity_f05(truth, pred, beta=beta)

    macro_score = float(np.mean(list(per_s1_scores.values()))) if per_s1_scores else 0.0
    return macro_score, per_s1_scores


def compute_recall_ceilings(
    s1_ids: Sequence[str],
    gt_map: Dict[str, Set[str]],
    union_candidates: Dict[str, Set[str]],
    statuses: Dict[str, str],
) -> Dict[str, Any]:
    """Compute overall and status-conditional candidate recall ceilings."""
    total_true = sum(len(gt_map[sid]) for sid in s1_ids)
    retrieved_true = sum(len(gt_map[sid] & union_candidates.get(sid, set())) for sid in s1_ids)
    overall_ceiling = float(retrieved_true / total_true) if total_true > 0 else 1.0

    # Matched in candidates subset
    mic_s1 = [sid for sid in s1_ids if statuses.get(sid) == "matched_in_candidates"]
    mic_true = sum(len(gt_map[sid]) for sid in mic_s1)
    mic_ret = sum(len(gt_map[sid] & union_candidates.get(sid, set())) for sid in mic_s1)
    mic_ceiling = float(mic_ret / mic_true) if mic_true > 0 else 0.0

    # Matched but missed subset
    mbm_s1 = [sid for sid in s1_ids if statuses.get(sid) == "matched_but_missed"]
    mbm_true = sum(len(gt_map[sid]) for sid in mbm_s1)

    return {
        "overall_candidate_ceiling_recall": overall_ceiling,
        "matched_in_candidates_ceiling_recall": mic_ceiling,
        "matched_but_missed_ceiling_recall": 0.0,
        "total_true_pairs": total_true,
        "retrieved_true_pairs": retrieved_true,
        "matched_in_candidates_true_pairs": mic_true,
        "matched_in_candidates_retrieved_pairs": mic_ret,
        "matched_but_missed_true_pairs": mbm_true,
    }


# =========================================================================
# 3. Entity-Level Stratified Train/Validation Split
# =========================================================================


def split_s1_entities(
    s1_ids: List[str],
    statuses: Dict[str, str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Deterministically split S1 entities 80/20 stratified by entity status.

    Never splits by candidate pair; strictly prevents any entity's candidates
    from appearing across both train and validation partitions.
    """
    status_list = [statuses[sid] for sid in s1_ids]
    from collections import Counter
    counts = Counter(status_list)
    can_stratify = all(c >= 2 for c in counts.values()) and len(counts) > 1

    if can_stratify:
        train_ids, val_ids = train_test_split(
            s1_ids,
            test_size=val_fraction,
            random_state=seed,
            stratify=status_list,
        )
    else:
        train_ids, val_ids = train_test_split(
            s1_ids,
            test_size=val_fraction,
            random_state=seed,
        )
    # Assert zero entity leakage
    assert set(train_ids).isdisjoint(set(val_ids)), "Train and validation S1 entities must be disjoint."
    return sorted(train_ids), sorted(val_ids)


# =========================================================================
# 4. Candidate Retrieval & Metadata Extraction
# =========================================================================


def get_or_build_vectorizers(
    s2_path: Path,
    s3_path: Path,
    cache_dir: Path,
    chunk_size: int = 100000,
    total_target_rows: int = 10320219,
    max_target_rows: Optional[int] = None,
    force_recompute: bool = False,
) -> Tuple[Any, Any]:
    """Load or construct unified global TfidfVectorizers across target chunks."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    char_cache_file = cache_dir / "char_vectorizer.joblib"
    word_cache_file = cache_dir / "word_vectorizer.joblib"

    if (
        not force_recompute
        and max_target_rows is None
        and char_cache_file.exists()
        and word_cache_file.exists()
    ):
        print(f"[Cache] Loading cached global TF-IDF vectorizers from {cache_dir}...")
        char_vec = joblib.load(char_cache_file)
        word_vec = joblib.load(word_cache_file)
        return char_vec, word_vec

    char_vec, word_vec = build_dual_consistent_vectorizers(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=chunk_size,
        total_target_rows=total_target_rows,
        max_target_rows=max_target_rows,
        char_ngram_range=(3, 5),
        word_ngram_range=WORD_NGRAM_RANGE,
    )

    if max_target_rows is None:
        print(f"[Cache] Saving global TF-IDF vectorizers to {cache_dir}...")
        joblib.dump(char_vec, char_cache_file)
        joblib.dump(word_vec, word_cache_file)

    return char_vec, word_vec


def generate_union_candidates(
    s1_sample_df: pd.DataFrame,
    s2_path: Path,
    s3_path: Path,
    gt_map: Dict[str, Set[str]],
    cache_dir: Path,
    char_top_k: int = CHAR_TFIDF_TOP_K,
    word_top_k: int = WORD_TFIDF_TOP_K,
    chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Set[str]], Dict[str, str]]:
    """Generate Character+Word UNION candidates against target corpus.

    Returns:
        candidate_pairs_df, char_scores_map, word_scores_map, union_candidates_map, statuses
    """
    sample_size = len(s1_sample_df)
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_cache_file = cache_dir / f"candidates_s{sample_size}_seed42_c{char_top_k}_w{word_top_k}.parquet"
    if not candidates_cache_file.exists() and char_top_k == 20 and word_top_k == 20:
        legacy_cache = cache_dir / f"candidates_s{sample_size}_seed42.parquet"
        if legacy_cache.exists():
            candidates_cache_file = legacy_cache

    if not force_recompute and max_target_rows is None and candidates_cache_file.exists():
        print(f"\n[Cache] Loading cached union candidate pairs from {candidates_cache_file}...")
        pairs_df = pd.read_parquet(candidates_cache_file)

        char_scores: Dict[str, Dict[str, float]] = {}
        word_scores: Dict[str, Dict[str, float]] = {}
        union_candidates: Dict[str, Set[str]] = {}

        for _, row in pairs_df.iterrows():
            sid = str(row["source1_entity_id"])
            cid = str(row["matched_entity_id"])
            cs = float(row["char_score"])
            ws = float(row["word_score"])

            if sid not in char_scores:
                char_scores[sid] = {}
                word_scores[sid] = {}
                union_candidates[sid] = set()

            if cs > 0.0:
                char_scores[sid][cid] = cs
            if ws > 0.0:
                word_scores[sid][cid] = ws
            union_candidates[sid].add(cid)

        # Classify statuses
        statuses: Dict[str, str] = {}
        for sid in s1_sample_df["entity_id"].astype(str):
            c_set = union_candidates.get(sid, set())
            statuses[sid] = classify_s1_status(gt_map.get(sid, set()), c_set)

        return pairs_df, char_scores, word_scores, union_candidates, statuses

    # Compute target row count
    s2_lines = sum(1 for _ in open(s2_path, "r", encoding="utf-8")) - 1
    s3_lines = sum(1 for _ in open(s3_path, "r", encoding="utf-8")) - 1
    total_target_rows = s2_lines + s3_lines
    eval_target_rows = total_target_rows if max_target_rows is None else min(max_target_rows, total_target_rows)
    num_chunks = int(np.ceil(eval_target_rows / chunk_size))

    # 1. Build or load unified vectorizers
    char_vec, word_vec = get_or_build_vectorizers(
        s2_path=s2_path,
        s3_path=s3_path,
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        total_target_rows=total_target_rows,
        max_target_rows=max_target_rows,
        force_recompute=force_recompute,
    )

    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_sample_df)

    print(f"\nTransforming {len(s1_ids):,} S1 queries into global TF-IDF feature spaces...")
    X_s1_char = char_vec.transform(s1_names).tocsr()
    X_s1_word = word_vec.transform(s1_names).tocsr()

    # Pass 2: Stream target chunks and perform memory-bounded dual retrieval
    print(f"\n[Pass 2/2] Streaming {eval_target_rows:,} target records across ~{num_chunks} chunks...")
    char_candidates: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    word_candidates: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}

    chunk_stream = stream_target_chunks(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=chunk_size,
        max_target_rows=max_target_rows,
    )

    t_pass2_start = time.time()
    total_processed = 0

    for chunk_idx, (target_chunk_df, source_label) in enumerate(chunk_stream, start=1):
        chunk_len = len(target_chunk_df)
        total_processed += chunk_len
        target_ids = target_chunk_df["entity_id"].astype(str).tolist()
        target_names = _extract_normalized_names(target_chunk_df)

        # 2a. Character TF-IDF
        X_chunk_char = char_vec.transform(target_names).tocsr()
        X_chunk_char_T = X_chunk_char.T.tocsr()

        for i, sid in enumerate(s1_ids):
            q_vec = X_s1_char[i : i + 1]
            if q_vec.nnz == 0:
                continue
            row_sims = q_vec.dot(X_chunk_char_T).tocsr()
            if row_sims.nnz == 0:
                continue

            data = row_sims.data
            cols = row_sims.indices
            if len(data) > char_top_k * 2:
                top_idx = np.argpartition(data, -char_top_k * 2)[-char_top_k * 2:]
                cols = cols[top_idx]
                data = data[top_idx]

            s1_map = char_candidates[sid]
            for col, score in zip(cols, data):
                cid = target_ids[col]
                s = float(score)
                if cid not in s1_map or s > s1_map[cid]:
                    s1_map[cid] = s

            if len(s1_map) > char_top_k * 4:
                top_items = heapq.nsmallest(
                    char_top_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                )
                char_candidates[sid] = dict(top_items)

        del X_chunk_char
        del X_chunk_char_T

        # 2b. Word TF-IDF
        X_chunk_word = word_vec.transform(target_names).tocsr()
        X_chunk_word_T = X_chunk_word.T.tocsr()

        for i, sid in enumerate(s1_ids):
            q_vec = X_s1_word[i : i + 1]
            if q_vec.nnz == 0:
                continue
            row_sims = q_vec.dot(X_chunk_word_T).tocsr()
            if row_sims.nnz == 0:
                continue

            data = row_sims.data
            cols = row_sims.indices
            if len(data) > word_top_k * 2:
                top_idx = np.argpartition(data, -word_top_k * 2)[-word_top_k * 2:]
                cols = cols[top_idx]
                data = data[top_idx]

            s1_map = word_candidates[sid]
            for col, score in zip(cols, data):
                cid = target_ids[col]
                s = float(score)
                if cid not in s1_map or s > s1_map[cid]:
                    s1_map[cid] = s

            if len(s1_map) > word_top_k * 4:
                top_items = heapq.nsmallest(
                    word_top_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                )
                word_candidates[sid] = dict(top_items)

        del X_chunk_word
        del X_chunk_word_T
        del target_chunk_df
        del target_ids
        del target_names
        gc.collect()

        if chunk_idx == 1 or chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            pct = (total_processed / eval_target_rows) * 100.0
            rss_mb = psutil.Process().memory_info().rss / 1e6
            el = time.time() - t_pass2_start
            print(f"  [Pass 2: Chunk {chunk_idx:>3}/{num_chunks}] Processed {total_processed:>10,} / {eval_target_rows:,} targets ({pct:>5.1f}%). Elapsed: {el:.1f}s | RSS: {rss_mb:.1f} MB")

    # Final Top-K extraction and candidate union formation
    final_char_scores: Dict[str, Dict[str, float]] = {}
    final_word_scores: Dict[str, Dict[str, float]] = {}
    final_union_candidates: Dict[str, Set[str]] = {}
    pair_rows: List[Dict[str, Any]] = []

    for sid in s1_ids:
        c_top = heapq.nsmallest(char_top_k, char_candidates[sid].items(), key=lambda x: (-x[1], x[0]))
        w_top = heapq.nsmallest(word_top_k, word_candidates[sid].items(), key=lambda x: (-x[1], x[0]))

        c_map = dict(c_top)
        w_map = dict(w_top)
        u_set = set(c_map.keys()) | set(w_map.keys())

        final_char_scores[sid] = c_map
        final_word_scores[sid] = w_map
        final_union_candidates[sid] = u_set

        gt_matches = gt_map.get(sid, set())
        for cid in sorted(u_set):
            cs = c_map.get(cid, 0.0)
            ws = w_map.get(cid, 0.0)
            is_match = 1 if cid in gt_matches else 0
            pair_rows.append({
                "source1_entity_id": sid,
                "matched_entity_id": cid,
                "char_score": float(cs),
                "word_score": float(ws),
                "label": int(is_match),
            })

    pairs_df = pd.DataFrame(pair_rows)

    if max_target_rows is None:
        print(f"\n[Cache] Persisting {len(pairs_df):,} candidate pairs to {candidates_cache_file}...")
        pairs_df.to_parquet(candidates_cache_file, index=False)

    # Status classification
    statuses: Dict[str, str] = {}
    for sid in s1_ids:
        c_set = final_union_candidates.get(sid, set())
        statuses[sid] = classify_s1_status(gt_map.get(sid, set()), c_set)

    return pairs_df, final_char_scores, final_word_scores, final_union_candidates, statuses


def collect_candidate_target_records(
    candidate_ids: Set[str],
    s2_path: Path,
    s3_path: Path,
    chunk_size: int = 200000,
) -> Dict[str, dict]:
    """Scan Source 2 and Source 3 files to retrieve business attributes for candidate target IDs."""
    print(f"\nCollecting candidate record metadata for {len(candidate_ids):,} unique target candidates...")
    t0 = time.time()
    target_dict: Dict[str, dict] = {}
    needed = set(candidate_ids)

    for src_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not path.exists():
            continue
        reader = pd.read_csv(
            path,
            sep="\t",
            chunksize=chunk_size,
            usecols=["entity_id", "business_name", "business_address", "country"],
        )
        for chunk in reader:
            matches = chunk[chunk["entity_id"].isin(needed)]
            if len(matches) > 0:
                for _, row in matches.iterrows():
                    eid = str(row["entity_id"])
                    target_dict[eid] = {
                        "entity_id": eid,
                        "business_name": row["business_name"],
                        "address": row["business_address"],
                        "country": row["country"],
                    }
                    needed.discard(eid)
            if not needed:
                break
        if not needed:
            break

    print(f"Collected {len(target_dict):,} candidate records in {time.time() - t0:.2f}s.")
    return target_dict


# =========================================================================
# 5. Threshold Grid Optimization
# =========================================================================


def search_optimal_threshold(
    val_s1_ids: List[str],
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    gt_map: Dict[str, Set[str]],
    thresholds: Optional[Sequence[float]] = None,
    beta: float = 0.5,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """Search grid from 0.05 to 0.95 and select threshold maximizing entity-level validation macro F0.5.

    Returns:
        best_threshold, best_metrics, grid_results_list
    """
    if thresholds is None:
        thresholds = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 19)]

    # Attach predicted probabilities to val_pairs DataFrame
    eval_df = val_pairs_df[["source1_entity_id", "matched_entity_id"]].copy()
    eval_df["prob"] = val_probs

    grid_results = []
    best_threshold = 0.5
    best_macro_f05 = -1.0
    best_metrics: Dict[str, Any] = {}

    for t in thresholds:
        pred_subset = eval_df[eval_df["prob"] >= t]

        # Group by S1 entity
        pred_map: Dict[str, Set[str]] = {sid: set() for sid in val_s1_ids}
        for sid, grp in pred_subset.groupby("source1_entity_id"):
            if sid in pred_map:
                pred_map[sid] = set(grp["matched_entity_id"].tolist())

        macro_f05, _ = evaluate_entity_macro_f05(val_s1_ids, gt_map, pred_map, beta=beta)

        # Entity prediction statistics
        n_s1_with_preds = sum(1 for sid in val_s1_ids if len(pred_map[sid]) > 0)
        total_pred_matches = sum(len(pred_map[sid]) for sid in val_s1_ids)
        avg_pred_matches = float(total_pred_matches / len(val_s1_ids)) if val_s1_ids else 0.0

        tp_count = sum(len(gt_map.get(sid, set()) & pred_map[sid]) for sid in val_s1_ids)
        fp_count = sum(len(pred_map[sid] - gt_map.get(sid, set())) for sid in val_s1_ids)
        fn_count = sum(len(gt_map.get(sid, set()) - pred_map[sid]) for sid in val_s1_ids)

        pair_precision = float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
        pair_recall = float(tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0.0

        entry = {
            "threshold": float(t),
            "macro_f05": float(macro_f05),
            "average_precision": float(pair_precision),
            "pair_recall": float(pair_recall),
            "number_of_s1_with_predictions": int(n_s1_with_preds),
            "avg_predicted_matches_per_s1": float(avg_pred_matches),
            "true_positive_count": int(tp_count),
            "false_positive_count": int(fp_count),
            "false_negative_count": int(fn_count),
        }
        grid_results.append(entry)

        # Maximize macro F0.5; break ties using precision, then threshold
        is_better = False
        if macro_f05 > best_macro_f05:
            is_better = True
        elif np.isclose(macro_f05, best_macro_f05, atol=1e-7):
            if pair_precision > best_metrics.get("average_precision", 0.0):
                is_better = True
            elif np.isclose(pair_precision, best_metrics.get("average_precision", 0.0), atol=1e-7) and t > best_threshold:
                is_better = True

        if is_better:
            best_macro_f05 = macro_f05
            best_threshold = float(t)
            best_metrics = entry

    return best_threshold, best_metrics, grid_results


# =========================================================================
# 6. End-to-End Training Pipeline Execution
# =========================================================================


def train_matcher(
    sample_size: int = 1000,
    seed: int = 42,
    val_fraction: float = 0.2,
    char_top_k: int = CHAR_TFIDF_TOP_K,
    word_top_k: int = WORD_TFIDF_TOP_K,
    model_output_filename: Optional[str] = None,
    metrics_output_filename: Optional[str] = None,
    target_chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
    force_recompute_candidates: bool = False,
    s1_path: Path = TRAIN_SOURCE1,
    s2_path: Path = TRAIN_SOURCE2,
    s3_path: Path = TRAIN_SOURCE3,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute the complete training baseline for the Business Entity Resolution matching stage."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION - TRAINING (K={char_top_k})")
    print("=" * 88)
    print(f"Sample size: {sample_size:,} S1 entities (seed: {seed})")
    print(f"Split: {int((1-val_fraction)*100)}% Train / {int(val_fraction*100)}% Validation (Stratified by S1 Status)")
    print(f"Candidate Retrieval: Character Top-K={char_top_k} UNION Word Top-K={word_top_k}")
    print(f"Output directory: {output_dir}")

    # 1. Sample Source 1 entities and load ground truth
    print(f"\n1. Sampling {sample_size:,} Source 1 entities...")
    s1_full_df = pd.read_csv(s1_path, sep="\t")
    s1_sample_df = s1_full_df.sample(n=min(sample_size, len(s1_full_df)), random_state=seed).copy()
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()

    if "business_address" in s1_sample_df.columns:
        s1_sample_df["address"] = s1_sample_df["business_address"]

    s1_dict = s1_sample_df.set_index("entity_id").to_dict(orient="index")

    print(f"2. Loading ground truth match sets for sampled S1 entities...")
    gt_map = load_ground_truth_for_s1(gt_path, set(s1_ids))
    total_true_pairs = sum(len(v) for v in gt_map.values())
    n_singletons_gt = sum(1 for v in gt_map.values() if len(v) == 0)
    print(f"   Found {total_true_pairs:,} true matches across {len(s1_ids) - n_singletons_gt:,} S1 entities ({n_singletons_gt:,} singletons).")

    # 2. Candidate Retrieval via Character+Word UNION
    print(f"\n3. Generating Character+Word UNION candidate pairs against target corpus (Top-K={char_top_k})...")
    pairs_df, char_scores, word_scores, union_cands, statuses = generate_union_candidates(
        s1_sample_df=s1_sample_df,
        s2_path=s2_path,
        s3_path=s3_path,
        gt_map=gt_map,
        cache_dir=cache_dir,
        char_top_k=char_top_k,
        word_top_k=word_top_k,
        chunk_size=target_chunk_size,
        max_target_rows=max_target_rows,
        force_recompute=force_recompute_candidates,
    )

    # 3. Report S1 Statuses and Recall Ceiling
    status_counts = {
        "singleton": sum(1 for s in statuses.values() if s == "singleton"),
        "matched_in_candidates": sum(1 for s in statuses.values() if s == "matched_in_candidates"),
        "matched_but_missed": sum(1 for s in statuses.values() if s == "matched_but_missed"),
    }
    print("\n" + "-" * 50)
    print("S1 ENTITY STATUS COUNTS:")
    print(f"  - singleton (zero-match):           {status_counts['singleton']:>5,}")
    print(f"  - matched_in_candidates:            {status_counts['matched_in_candidates']:>5,}")
    print(f"  - matched_but_missed:               {status_counts['matched_but_missed']:>5,}")
    print("-" * 50)

    recall_ceilings = compute_recall_ceilings(s1_ids, gt_map, union_cands, statuses)
    print(f"CANDIDATE RECALL CEILING:")
    print(f"  - Overall Ceiling Recall:           {recall_ceilings['overall_candidate_ceiling_recall']*100:.2f}% ({recall_ceilings['retrieved_true_pairs']:,} / {recall_ceilings['total_true_pairs']:,} true pairs)")
    print(f"  - Matched-in-Candidates Recall:     {recall_ceilings['matched_in_candidates_ceiling_recall']*100:.2f}%")
    print(f"  - Average Candidates per S1:        {len(pairs_df) / len(s1_ids):.2f}")
    print("-" * 50)

    # 4. Stratified Split by S1 Entity (NEVER by candidate pair)
    print("\n4. Splitting S1 entities into Train / Validation (stratified by status)...")
    train_s1_ids, val_s1_ids = split_s1_entities(
        s1_ids=s1_ids,
        statuses=statuses,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    train_pairs_df = pairs_df[pairs_df["source1_entity_id"].isin(train_s1_set)].copy().reset_index(drop=True)
    val_pairs_df = pairs_df[pairs_df["source1_entity_id"].isin(val_s1_set)].copy().reset_index(drop=True)

    print(f"   Train partition: {len(train_s1_ids):,} S1 entities ({len(train_pairs_df):,} candidate pairs, {train_pairs_df['label'].sum():,} positives)")
    print(f"   Val partition:   {len(val_s1_ids):,} S1 entities ({len(val_pairs_df):,} candidate pairs, {val_pairs_df['label'].sum():,} positives)")

    # 5. Extract target metadata and build pairwise feature matrices
    print("\n5. Extracting candidate target record metadata...")
    all_candidate_ids = set(pairs_df["matched_entity_id"].astype(str).unique())
    target_dict = collect_candidate_target_records(all_candidate_ids, s2_path, s3_path)

    print("   Generating pairwise features using src/features.py...")
    t_feat = time.time()
    feature_matrix_df = build_feature_matrix(
        candidate_pairs=pairs_df[["source1_entity_id", "matched_entity_id", "char_score", "word_score"]],
        s1_lookup=s1_dict,
        target_lookup=target_dict,
        chunk_size=5000,
    )
    print(f"   Computed {len(FEATURE_COLUMNS)} features for {len(feature_matrix_df):,} pairs in {time.time() - t_feat:.2f}s.")

    # Split feature matrix to train and val
    train_mask = feature_matrix_df["source1_entity_id"].isin(train_s1_set)
    val_mask = feature_matrix_df["source1_entity_id"].isin(val_s1_set)

    X_train = feature_matrix_df.loc[train_mask, FEATURE_COLUMNS].copy()
    y_train = train_pairs_df["label"].values.astype(int)

    X_val = feature_matrix_df.loc[val_mask, FEATURE_COLUMNS].copy()
    y_val = val_pairs_df["label"].values.astype(int)

    # 6. Train Downstream LightGBM Classifier
    print("\n6. Training LightGBM binary classifier with early stopping...")
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": seed,
        "n_estimators": 250,
        "n_jobs": -1,
        "verbose": -1,
    }
    if LGBM_PARAMS:
        base_params.update(LGBM_PARAMS)
    base_params["random_state"] = seed
    base_params["class_weight"] = "balanced"

    model = lgb.LGBMClassifier(**base_params)

    # PR-AUC custom evaluation metric for early stopping
    def pr_auc_eval(y_true, y_pred):
        if np.sum(y_true) == 0:
            return "pr_auc", 0.0, True
        score = average_precision_score(y_true, y_pred)
        return "pr_auc", score, True

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        warnings.filterwarnings("ignore", message=".*No positive class found.*")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["binary_logloss", pr_auc_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )

    print(f"   Model fitted successfully. Best iteration: {model.best_iteration_}")
    val_probs = model.predict_proba(X_val)[:, 1]
    if np.sum(y_val) > 0:
        val_pr_auc = float(average_precision_score(y_val, val_probs))
    else:
        val_pr_auc = 0.0
    print(f"   Validation PR-AUC (Average Precision): {val_pr_auc:.4f}")

    # 7. Entity-Level Decision Threshold Optimization for Macro F0.5
    print("\n7. Searching decision threshold grid (0.05 - 0.95) for Macro F0.5...")
    best_threshold, best_metrics, grid_results = search_optimal_threshold(
        val_s1_ids=val_s1_ids,
        val_pairs_df=val_pairs_df,
        val_probs=val_probs,
        gt_map=gt_map,
        beta=0.5,
    )

    # 8. Compute Detailed Validation Macro Scores at Best Threshold
    val_eval_df = val_pairs_df[["source1_entity_id", "matched_entity_id"]].copy()
    val_eval_df["prob"] = val_probs
    val_pred_subset = val_eval_df[val_eval_df["prob"] >= best_threshold]

    val_pred_map: Dict[str, Set[str]] = {sid: set() for sid in val_s1_ids}
    for sid, grp in val_pred_subset.groupby("source1_entity_id"):
        if sid in val_pred_map:
            val_pred_map[sid] = set(grp["matched_entity_id"].tolist())

    end_to_end_macro_f05, per_s1_f05 = evaluate_entity_macro_f05(val_s1_ids, gt_map, val_pred_map, beta=0.5)

    # Matcher-conditional macro F0.5 (evaluated only on S1 where at least one GT match was retrieved)
    mic_val_s1 = [sid for sid in val_s1_ids if statuses.get(sid) == "matched_in_candidates"]
    mic_macro_f05 = float(np.mean([per_s1_f05[sid] for sid in mic_val_s1])) if mic_val_s1 else 0.0

    val_ceiling_info = compute_recall_ceilings(val_s1_ids, gt_map, union_cands, statuses)

    total_runtime = time.time() - t_start

    print("\n" + "=" * 88)
    print("TRAINING & VALIDATION RESULTS SUMMARY:")
    print("=" * 88)
    print(f"Top-K per Channel:                         {char_top_k}")
    print(f"Selected Decision Threshold:               {best_threshold:.4f}")
    print(f"Validation End-to-End Macro F0.5:          {end_to_end_macro_f05:.4f}")
    print(f"Validation Matcher-Conditional Macro F0.5: {mic_macro_f05:.4f}")
    print(f"Validation Candidate Ceiling Recall:       {val_ceiling_info['overall_candidate_ceiling_recall']*100:.2f}%")
    print(f"Validation Average Precision (PR-AUC):     {val_pr_auc:.4f}")
    print(f"Validation Pair Precision at Threshold:    {best_metrics['average_precision']:.4f}")
    print(f"Validation Pair Recall at Threshold:       {best_metrics['pair_recall']:.4f}")
    peak_memory_mb = get_peak_memory_mb()
    print(f"Total Runtime:                             {total_runtime:.2f}s")
    print(f"Peak Memory (RSS):                         {peak_memory_mb:.2f} MB")
    print("=" * 88)

    # 9. Save Artifacts
    if model_output_filename is not None:
        model_txt_path = output_dir / model_output_filename
    else:
        model_txt_path = output_dir / ("matcher_model.txt" if char_top_k == 20 else f"matcher_model_k{char_top_k}.txt")

    if metrics_output_filename is not None:
        metrics_json_path = output_dir / metrics_output_filename
    else:
        metrics_json_path = output_dir / ("training_metrics.json" if char_top_k == 20 else f"training_metrics_k{char_top_k}.json")

    print(f"\nSaving LightGBM booster model to {model_txt_path}...")
    model.booster_.save_model(str(model_txt_path))

    print(f"Saving training metrics and experiment metadata to {metrics_json_path}...")

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sample_size": sample_size,
        "target_corpus_size": 10320219 if max_target_rows is None else max_target_rows,
        "blocking_configuration": {
            "char_top_k": char_top_k,
            "word_top_k": word_top_k,
            "char_ngram_range": [3, 5],
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "target_chunk_size": target_chunk_size,
        },
        "status_counts": status_counts,
        "candidate_counts": {
            "total_candidate_pairs": len(pairs_df),
            "avg_candidates_per_s1": float(len(pairs_df) / len(s1_ids)),
        },
        "candidate_recall_ceiling": recall_ceilings,
        "split_information": {
            "train_s1_count": len(train_s1_ids),
            "val_s1_count": len(val_s1_ids),
            "train_pairs_count": len(train_pairs_df),
            "val_pairs_count": len(val_pairs_df),
            "train_positives": int(train_pairs_df["label"].sum()),
            "val_positives": int(val_pairs_df["label"].sum()),
            "val_fraction": val_fraction,
        },
        "selected_threshold": best_threshold,
        "validation_metrics": {
            "end_to_end_macro_f05": end_to_end_macro_f05,
            "matcher_conditional_macro_f05": mic_macro_f05,
            "candidate_ceiling_recall": val_ceiling_info["overall_candidate_ceiling_recall"],
            "val_pr_auc": val_pr_auc,
            "precision_at_threshold": best_metrics["average_precision"],
            "recall_at_threshold": best_metrics["pair_recall"],
            "true_positive_count": best_metrics["true_positive_count"],
            "false_positive_count": best_metrics["false_positive_count"],
            "false_negative_count": best_metrics["false_negative_count"],
            "number_of_s1_with_predictions": best_metrics["number_of_s1_with_predictions"],
            "avg_predicted_matches_per_s1": best_metrics["avg_predicted_matches_per_s1"],
        },
        "lightgbm_parameters": {
            k: v for k, v in base_params.items() if not callable(v)
        },
        "threshold_grid_results": grid_results,
        "training_runtime_seconds": float(round(total_runtime, 2)),
        "peak_memory_mb": float(round(peak_memory_mb, 2)),
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Business Entity Resolution Matcher Baseline.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of S1 training queries (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and model training (default: 42).")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation partition fraction (default: 0.2).")
    parser.add_argument("--char-top-k", type=int, default=CHAR_TFIDF_TOP_K, help=f"Character Top-K cutoff (default: {CHAR_TFIDF_TOP_K}).")
    parser.add_argument("--word-top-k", type=int, default=WORD_TFIDF_TOP_K, help=f"Word Top-K cutoff (default: {WORD_TFIDF_TOP_K}).")
    parser.add_argument("--model-output", type=str, default=None, help="Booster model output filename.")
    parser.add_argument("--metrics-output", type=str, default=None, help="Metrics JSON output filename.")
    parser.add_argument("--target-chunk-size", type=int, default=100000, help="Target streaming chunk size (default: 100000).")
    parser.add_argument("--max-target-rows", type=int, default=None, help="Optional max target rows for dry runs.")
    parser.add_argument("--recompute-candidates", action="store_true", help="Force recomputing candidate retrieval.")
    args = parser.parse_args()

    train_matcher(
        sample_size=args.sample_size,
        seed=args.seed,
        val_fraction=args.val_fraction,
        char_top_k=args.char_top_k,
        word_top_k=args.word_top_k,
        model_output_filename=args.model_output,
        metrics_output_filename=args.metrics_output,
        target_chunk_size=args.target_chunk_size,
        max_target_rows=args.max_target_rows,
        force_recompute_candidates=args.recompute_candidates,
    )


if __name__ == "__main__":
    main()
