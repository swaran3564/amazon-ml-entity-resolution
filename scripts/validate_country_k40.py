"""Validate country-partitioned candidate retrieval on 1,000 S1 validation split.

Re-evaluates the existing approved K=40 model (output/matcher_model_k40.txt)
with dynamic country-partitioned candidate generation across the full
10,320,219-row target corpus.
"""

import gc
import heapq
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_union_blocking import load_ground_truth_for_s1
from scripts.run_test_inference import stream_target_chunks_with_country
from scripts.train_matcher import (
    assign_candidate_labels,
    classify_s1_status,
    collect_candidate_target_records,
    compute_recall_ceilings,
    evaluate_entity_macro_f05,
    get_peak_memory_mb,
    search_optimal_threshold,
    split_s1_entities,
)
from src.blocking import _extract_normalized_names
from src.config import (
    CHAR_TFIDF_TOP_K,
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    WORD_NGRAM_RANGE,
    WORD_TFIDF_TOP_K,
)
from src.features import FEATURE_COLUMNS, build_feature_matrix


def normalize_country(country: Any) -> str:
    """Safely normalize a country string via strip and casefold.

    Returns '' for missing/empty/null/NaN representations.
    """
    if country is None or pd.isna(country):
        return ""
    s = str(country).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s.casefold()


def retrieve_country_partitioned_candidates(
    s1_sample_df: pd.DataFrame,
    s2_path: Path,
    s3_path: Path,
    cache_dir: Path,
    char_top_k: int = 40,
    word_top_k: int = 40,
    target_chunk_size: int = 100000,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Set[str]], Dict[str, str]]:
    """Retrieve Character Top-K and Word Top-K candidates using normalized country partitioning."""
    sample_size = len(s1_sample_df)
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_sample_df)
    s1_raw_countries = s1_sample_df["country"].tolist()
    s1_norm_countries = [normalize_country(c) for c in s1_raw_countries]

    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_cache_file = cache_dir / f"candidates_s{sample_size}_seed42_c{char_top_k}_w{word_top_k}_country.parquet"

    # Load ground truth for status classification
    gt_map = load_ground_truth_for_s1(TRAIN_GROUND_TRUTH, set(s1_ids))

    if not force_recompute and candidates_cache_file.exists():
        print(f"\n[Cache] Loading cached country-partitioned candidate pairs from {candidates_cache_file}...")
        pairs_df = pd.read_parquet(candidates_cache_file)

        char_scores: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
        word_scores: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
        union_candidates: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

        for _, row in pairs_df.iterrows():
            sid = str(row["source1_entity_id"])
            cid = str(row["matched_entity_id"])
            cs = float(row["char_score"])
            ws = float(row["word_score"])
            if cs > 0.0:
                char_scores[sid][cid] = cs
            if ws > 0.0:
                word_scores[sid][cid] = ws
            union_candidates[sid].add(cid)

        statuses: Dict[str, str] = {}
        for sid in s1_ids:
            c_set = union_candidates.get(sid, set())
            statuses[sid] = classify_s1_status(gt_map.get(sid, set()), c_set)

        return pairs_df, char_scores, word_scores, union_candidates, statuses

    # Group S1 queries by normalized country
    queries_by_country: Dict[str, List[Tuple[int, str]]] = {}
    fallback_queries: List[Tuple[int, str]] = []  # S1 with empty/unusable country

    for idx, (sid, norm_c) in enumerate(zip(s1_ids, s1_norm_countries)):
        if norm_c == "":
            fallback_queries.append((idx, sid))
        else:
            queries_by_country.setdefault(norm_c, []).append((idx, sid))

    print(f"\nS1 Country Groups: { {c: len(l) for c, l in queries_by_country.items()} } (Fallback: {len(fallback_queries)})")

    # Load pre-fitted vectorizers
    char_vec_path = cache_dir / "char_vectorizer.joblib"
    word_vec_path = cache_dir / "word_vectorizer.joblib"
    print(f"Loading vectorizers from {cache_dir}...")
    char_vec = joblib.load(char_vec_path)
    word_vec = joblib.load(word_vec_path)

    # Vectorize queries
    print("Transforming S1 queries into TF-IDF spaces...")
    X_s1_char = char_vec.transform(s1_names).tocsr()
    X_s1_word = word_vec.transform(s1_names).tocsr()

    char_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    word_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}

    chunk_stream = stream_target_chunks_with_country(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=target_chunk_size,
    )

    t_pass2_start = time.time()
    total_processed = 0

    print(f"Streaming target records for country-partitioned retrieval...")
    for chunk_idx, (chunk_df, source_label) in enumerate(chunk_stream, start=1):
        chunk_len = len(chunk_df)
        total_processed += chunk_len

        target_ids = np.array(chunk_df["entity_id"].astype(str).tolist())
        target_names = _extract_normalized_names(chunk_df)
        target_norm_countries = np.array([normalize_country(c) for c in chunk_df["country"].tolist()])

        X_chunk_char = char_vec.transform(target_names).tocsr()
        X_chunk_word = word_vec.transform(target_names).tocsr()

        unique_countries = np.unique(target_norm_countries)

        # 1. Standard country partitions
        for c_val in unique_countries:
            if c_val == "":
                continue  # handled in fallback

            if c_val not in queries_by_country and not fallback_queries:
                continue

            c_mask = (target_norm_countries == c_val)
            c_target_ids = target_ids[c_mask]
            X_c_char_T = X_chunk_char[c_mask].T.tocsr()
            X_c_word_T = X_chunk_word[c_mask].T.tocsr()

            # Queries matching this country + any fallback queries
            eval_queries = queries_by_country.get(c_val, []) + fallback_queries

            for orig_idx, sid in eval_queries:
                # Char
                q_char = X_s1_char[orig_idx : orig_idx + 1]
                if q_char.nnz > 0:
                    sims = q_char.dot(X_c_char_T).tocsr()
                    if sims.nnz > 0:
                        data, cols = sims.data, sims.indices
                        if len(data) > char_top_k * 2:
                            top_idx = np.argpartition(data, -char_top_k * 2)[-char_top_k * 2:]
                            cols, data = cols[top_idx], data[top_idx]
                        s1_map = char_cands[sid]
                        for col, score in zip(cols, data):
                            cid = c_target_ids[col]
                            s = float(score)
                            if cid not in s1_map or s > s1_map[cid]:
                                s1_map[cid] = s
                        if len(s1_map) > char_top_k * 4:
                            top_items = heapq.nsmallest(
                                char_top_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                            )
                            char_cands[sid] = dict(top_items)

                # Word
                q_word = X_s1_word[orig_idx : orig_idx + 1]
                if q_word.nnz > 0:
                    sims = q_word.dot(X_c_word_T).tocsr()
                    if sims.nnz > 0:
                        data, cols = sims.data, sims.indices
                        if len(data) > word_top_k * 2:
                            top_idx = np.argpartition(data, -word_top_k * 2)[-word_top_k * 2:]
                            cols, data = cols[top_idx], data[top_idx]
                        s1_map = word_cands[sid]
                        for col, score in zip(cols, data):
                            cid = c_target_ids[col]
                            s = float(score)
                            if cid not in s1_map or s > s1_map[cid]:
                                s1_map[cid] = s
                        if len(s1_map) > word_top_k * 4:
                            top_items = heapq.nsmallest(
                                word_top_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                            )
                            word_cands[sid] = dict(top_items)

        # 2. Target Fallback: any target with empty country is evaluated against ALL S1 queries
        if "" in unique_countries:
            fb_mask = (target_norm_countries == "")
            fb_target_ids = target_ids[fb_mask]
            X_fb_char_T = X_chunk_char[fb_mask].T.tocsr()
            X_fb_word_T = X_chunk_word[fb_mask].T.tocsr()

            for orig_idx, sid in enumerate(s1_ids):
                # Char
                q_char = X_s1_char[orig_idx : orig_idx + 1]
                if q_char.nnz > 0:
                    sims = q_char.dot(X_fb_char_T).tocsr()
                    if sims.nnz > 0:
                        data, cols = sims.data, sims.indices
                        s1_map = char_cands[sid]
                        for col, score in zip(cols, data):
                            cid = fb_target_ids[col]
                            s = float(score)
                            if cid not in s1_map or s > s1_map[cid]:
                                s1_map[cid] = s
                # Word
                q_word = X_s1_word[orig_idx : orig_idx + 1]
                if q_word.nnz > 0:
                    sims = q_word.dot(X_fb_word_T).tocsr()
                    if sims.nnz > 0:
                        data, cols = sims.data, sims.indices
                        s1_map = word_cands[sid]
                        for col, score in zip(cols, data):
                            cid = fb_target_ids[col]
                            s = float(score)
                            if cid not in s1_map or s > s1_map[cid]:
                                s1_map[cid] = s

        del X_chunk_char
        del X_chunk_word
        del chunk_df
        gc.collect()

        if chunk_idx == 1 or chunk_idx % 20 == 0 or chunk_idx == 104:
            el = time.time() - t_pass2_start
            rss = psutil.Process().memory_info().rss / 1e6
            print(f"  [Chunk {chunk_idx:>3}/104] Processed {total_processed:>10,} targets. Elapsed: {el:.1f}s | RSS: {rss:.1f} MB")

    # Form final UNION candidates
    final_char_scores: Dict[str, Dict[str, float]] = {}
    final_word_scores: Dict[str, Dict[str, float]] = {}
    final_union_candidates: Dict[str, Set[str]] = {}
    pair_rows: List[Dict[str, Any]] = []

    for sid in s1_ids:
        c_top = heapq.nsmallest(char_top_k, char_cands[sid].items(), key=lambda x: (-x[1], x[0]))
        w_top = heapq.nsmallest(word_top_k, word_cands[sid].items(), key=lambda x: (-x[1], x[0]))

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
    print(f"\nPersisting {len(pairs_df):,} country-partitioned pairs to {candidates_cache_file}...")
    pairs_df.to_parquet(candidates_cache_file, index=False)

    statuses: Dict[str, str] = {}
    for sid in s1_ids:
        c_set = final_union_candidates.get(sid, set())
        statuses[sid] = classify_s1_status(gt_map.get(sid, set()), c_set)

    return pairs_df, final_char_scores, final_word_scores, final_union_candidates, statuses


def run_country_validation():
    t_start = time.time()
    sample_size = 1000
    seed = 42
    val_fraction = 0.2
    char_top_k = 40
    word_top_k = 40
    model_path = OUTPUT_DIR / "matcher_model_k40.txt"
    cache_dir = OUTPUT_DIR / "cache"

    print("=" * 88)
    print("TASK 3: RE-RUN 1,000-S1 VALIDATION WITH COUNTRY PARTITIONING")
    print("=" * 88)
    print(f"Sample Size:            {sample_size} S1 entities (seed={seed})")
    print(f"Candidate Retrieval:    Char Top-{char_top_k} UNION Word Top-{word_top_k} (Country-Partitioned)")
    print(f"Matcher Model:          {model_path}")
    print(f"Evaluated Thresholds:   [0.80, 0.85, 0.90, 0.95]")
    print("=" * 88)

    # 1. Sample 1,000 S1 queries
    s1_full_df = pd.read_csv(TRAIN_SOURCE1, sep="\t")
    s1_sample_df = s1_full_df.sample(n=sample_size, random_state=seed).copy()
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()

    if "business_address" in s1_sample_df.columns:
        s1_sample_df["address"] = s1_sample_df["business_address"]
    s1_dict = s1_sample_df.set_index("entity_id").to_dict(orient="index")

    gt_map = load_ground_truth_for_s1(TRAIN_GROUND_TRUTH, set(s1_ids))
    total_true_pairs = sum(len(m) for m in gt_map.values())
    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    print(f"Sampled {len(s1_ids)} S1 entities with {total_true_pairs} true pairs ({n_singletons} singletons).")

    # 2. Retrieve candidates with country partitioning
    pairs_df, char_scores, word_scores, union_cands, statuses = retrieve_country_partitioned_candidates(
        s1_sample_df=s1_sample_df,
        s2_path=TRAIN_SOURCE2,
        s3_path=TRAIN_SOURCE3,
        cache_dir=cache_dir,
        char_top_k=char_top_k,
        word_top_k=word_top_k,
    )

    status_counts = pd.Series(list(statuses.values())).value_counts().to_dict()
    recall_ceilings = compute_recall_ceilings(s1_ids, gt_map, union_cands, statuses)

    print("\n--------------------------------------------------")
    print("S1 ENTITY STATUS COUNTS (COUNTRY PARTITIONED):")
    for st, cnt in sorted(status_counts.items()):
        print(f"  - {st:<25}: {cnt:>6}")
    print("--------------------------------------------------")
    print("CANDIDATE RECALL CEILING:")
    print(f"  - Overall Ceiling Recall:       {recall_ceilings['overall_candidate_ceiling_recall']*100:.2f}% ({recall_ceilings['retrieved_true_pairs']}/{recall_ceilings['total_true_pairs']} true pairs)")
    print(f"  - Matched-in-Candidates Recall: {recall_ceilings['matched_in_candidates_ceiling_recall']*100:.2f}%")
    print(f"  - Total Candidate Pairs:        {len(pairs_df):,}")
    print(f"  - Average Candidates per S1:    {len(pairs_df)/len(s1_ids):.2f}")
    print("--------------------------------------------------")

    # 3. Stratified 80/20 train/validation split (same seed 42)
    train_s1_ids, val_s1_ids = split_s1_entities(s1_ids, statuses, val_fraction=val_fraction, seed=seed)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    train_pairs_df = pairs_df[pairs_df["source1_entity_id"].isin(train_s1_set)].copy()
    val_pairs_df = pairs_df[pairs_df["source1_entity_id"].isin(val_s1_set)].copy()

    # 4. Target record metadata lookup
    all_candidate_ids = set(pairs_df["matched_entity_id"].astype(str).unique())
    target_dict = collect_candidate_target_records(all_candidate_ids, TRAIN_SOURCE2, TRAIN_SOURCE3)

    # 5. Build features
    print("\nGenerating pairwise features using src/features.py...")
    t_feat = time.time()
    feature_matrix_df = build_feature_matrix(
        candidate_pairs=pairs_df[["source1_entity_id", "matched_entity_id", "char_score", "word_score"]],
        s1_lookup=s1_dict,
        target_lookup=target_dict,
        chunk_size=5000,
    )
    print(f"Computed features in {time.time() - t_feat:.2f}s.")

    # 6. Load approved model and predict validation probabilities
    print(f"\nLoading approved K=40 booster model from {model_path}...")
    booster = lgb.Booster(model_file=str(model_path))

    val_mask = feature_matrix_df["source1_entity_id"].isin(val_s1_set)
    X_val = feature_matrix_df.loc[val_mask, FEATURE_COLUMNS].copy()
    y_val = val_pairs_df["label"].values.astype(int)

    val_probs = booster.predict(X_val.values)
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    print(f"Validation PR-AUC (Average Precision): {val_pr_auc:.4f}")

    # 7. Evaluate thresholds 0.80, 0.85, 0.90, 0.95 and full grid
    mic_val_s1_ids = [sid for sid in val_s1_ids if statuses.get(sid) == "matched_in_candidates"]

    grid_thresholds = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 19)]
    target_eval_thresholds = [0.80, 0.85, 0.90, 0.95]

    best_threshold, best_metrics, grid_results = search_optimal_threshold(
        val_s1_ids=val_s1_ids,
        val_pairs_df=val_pairs_df,
        val_probs=val_probs,
        gt_map=gt_map,
        thresholds=grid_thresholds,
    )

    eval_df = val_pairs_df[["source1_entity_id", "matched_entity_id"]].copy()
    eval_df["prob"] = val_probs

    eval_results = {}
    for entry in grid_results:
        t = round(float(entry["threshold"]), 2)
        # Compute matcher conditional macro F0.5
        pred_subset = eval_df[eval_df["prob"] >= t]
        pred_map = {sid: set() for sid in mic_val_s1_ids}
        for sid, grp in pred_subset.groupby("source1_entity_id"):
            if sid in pred_map:
                pred_map[sid] = set(grp["matched_entity_id"].tolist())
        mic_score, _ = evaluate_entity_macro_f05(mic_val_s1_ids, gt_map, pred_map, beta=0.5)
        entry["matcher_conditional_macro_f05"] = float(mic_score)
        eval_results[t] = entry

    total_runtime = time.time() - t_start
    peak_rss = get_peak_memory_mb()

    # Print requested threshold report
    print("\n" + "=" * 96)
    print("THRESHOLD EVALUATION REPORT (COUNTRY-PARTITIONED CANDIDATES):")
    print("=" * 96)
    print(f"{'Threshold':<12} {'Macro F0.5':<14} {'Matcher-Cond F0.5':<20} {'Pair Precision':<16} {'Pair Recall':<14} {'Pred Matches/S1':<16}")
    print("-" * 96)
    for thr in target_eval_thresholds:
        m = eval_results[thr]
        print(f"{thr:<12.2f} {m['macro_f05']:<14.4f} {m['matcher_conditional_macro_f05']:<20.4f} {m['average_precision']:<16.4f} {m['pair_recall']:<14.4f} {m['avg_predicted_matches_per_s1']:<16.3f}")
    print("-" * 96)

    # Find overall best threshold on full grid
    best_thr = max(eval_results.keys(), key=lambda t: eval_results[t]["macro_f05"])
    best_m = eval_results[best_thr]
    print(f"\nBest Threshold Across Full Grid (0.05 - 0.95): {best_thr:.2f} (Macro F0.5 = {best_m['macro_f05']:.4f})")
    print(f"Total Runtime: {total_runtime:.2f}s | Peak RSS: {peak_rss:.2f} MB")
    print("=" * 96)

    # Save to country_validation_k40.json (do NOT overwrite training_metrics_k40.json)
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size": sample_size,
        "seed": seed,
        "status_counts": status_counts,
        "candidate_recall_ceiling": recall_ceilings,
        "total_candidate_pairs": len(pairs_df),
        "val_pr_auc": val_pr_auc,
        "best_threshold": best_thr,
        "threshold_evaluations": {
            str(thr): eval_results[thr] for thr in target_eval_thresholds
        },
        "full_grid_evaluations": {
            str(thr): eval_results[thr] for thr in grid_thresholds
        },
        "total_runtime_seconds": total_runtime,
        "peak_memory_mb": peak_rss,
    }

    out_file = OUTPUT_DIR / "country_validation_k40.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved country validation metrics to {out_file}")

    return out_payload


if __name__ == "__main__":
    run_country_validation()
