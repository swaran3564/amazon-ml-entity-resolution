"""Evaluate Word TF-IDF candidate blocking performance on training data.

This script measures real retrieval recall and candidate efficiency on a deterministic
sample of Source 1 entities against the FULL target candidate corpus (Source 2 + Source 3,
10,320,219 total records) using a mathematically valid two-pass streaming approach.

MATHEMATICAL RATIONALE FOR TWO-PASS CONSISTENT TF-IDF RETRIEVAL:
-----------------------------------------------------------------
In candidate blocking across a 10.3M-row target corpus, materializing the full TF-IDF
matrix in RAM requires >20 GB, exceeding typical local machine memory. However, fitting
a new TfidfVectorizer on each target chunk independently causes vocabulary misalignment
and disparate IDF weighting across chunks:
    IDF_chunk(t) = ln((N_chunk + 1) / (DF_chunk(t) + 1)) + 1
Cosine similarity scores computed against different local models cannot be directly
compared or globally merged into Top-K rankings.

To solve this while strictly bounding memory:
1. Pass 1 (Streaming Vocabulary & Document Frequency Accumulation):
   Streams target records in bounded chunks (default 100,000 rows). Accumulates corpus-wide
   document frequencies for word (1, 2) n-grams in a lightweight dictionary (~400 MB RAM
   for ~3M unique n-grams). The full target text or matrix is never held in memory.
2. Global Model Construction:
   Using total corpus size N = 10,320,219 and exact global document frequencies DF, computes
   the true corpus-wide IDF weights:
       IDF_global(t) = ln((N + 1) / (DF(t) + 1)) + 1
   Instantiates a single frozen TfidfVectorizer with the unified vocabulary and IDF vector.
3. Pass 2 (Streaming Candidate Retrieval & Global Top-K Merging):
   S1 queries are transformed once into this unified feature space. Target chunks are streamed
   and transformed using the SAME global vectorizer. Every dot product produces cosine similarities
   in the exact same coordinate space with identical IDF weights and L2 unit norms.
   Scores from all 104 chunks are directly comparable, enabling mathematically valid global
   Top-K candidate selection with deterministic tie-breaking across the full 10.32M target corpus
   in ~1.2 GB peak RAM.

Usage:
    python3 scripts/evaluate_word_blocking.py --sample-size 100
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
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocking import _extract_normalized_names, retrieve_word_tfidf_candidates
from src.config import (
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    WORD_NGRAM_RANGE,
    WORD_TFIDF_TOP_K,
)


def get_peak_memory_mb() -> float:
    """Get peak process resident set size (RSS) in megabytes."""
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw_rss / (1024 * 1024))
    return float(raw_rss / 1024)


def estimate_computation_safety(
    target_chunk_size: int,
    available_ram_bytes: Optional[int] = None,
) -> Tuple[bool, float, float, str]:
    """Estimate whether two-pass word TF-IDF retrieval is computationally safe on local hardware.

    Parameters
    ----------
    target_chunk_size : int
        Number of target candidate entities processed per chunk.
    available_ram_bytes : Optional[int]
        System available RAM in bytes (defaults to psutil.virtual_memory().available).

    Returns
    -------
    Tuple[bool, float, float, str]
        (is_safe, estimated_required_gb, available_gb, diagnostic_message)
    """
    if available_ram_bytes is None:
        try:
            available_ram_bytes = psutil.virtual_memory().available
        except Exception:
            available_ram_bytes = 8 * (1024**3)

    available_gb = available_ram_bytes / (1024**3)
    total_gb = psutil.virtual_memory().total / (1024**3)

    # Base memory: Python process (~0.4 GB) + global vocabulary dict (~0.5 GB) + per-chunk buffer (~0.3 GB)
    estimated_required_gb = 1.2 + (target_chunk_size / 1_000_000.0) * 1.2

    # Safe operating margin: estimated RAM must fit within 75% of currently available free RAM
    is_safe = estimated_required_gb <= (available_gb * 0.75)

    msg = (
        f"Memory Safety Assessment (Two-Pass Streaming Mode):\n"
        f"  - Target chunk size: {target_chunk_size:,} rows\n"
        f"  - Estimated RAM required: {estimated_required_gb:.2f} GB\n"
        f"  - Available system RAM: {available_gb:.2f} GB (Total physical RAM: {total_gb:.2f} GB)\n"
    )

    if is_safe:
        msg += f"  - Status: SAFE. Estimated memory fits comfortably within available RAM."
    else:
        msg += (
            f"  - Status: [UNSAFE - CONSTRAINT DETECTED].\n"
            f"    Estimated memory ({estimated_required_gb:.2f} GB) exceeds safe threshold of available RAM ({available_gb:.2f} GB free).\n"
            f"    Decrease --target-chunk-size (e.g. 50000) to proceed safely."
        )

    return is_safe, estimated_required_gb, available_gb, msg


def verify_consistent_scoring_and_merging() -> bool:
    """Synthetic verification: ensure separate chunks produce scores in the SAME feature space.

    Demonstrates that:
    1. Two separate chunks scored against a query produce bitwise identical scores for identical texts.
    2. Candidate scores from different chunks are directly comparable and correctly ordered by global Top-K.
    3. Deterministic secondary tie-breaking on entity_id is strictly preserved.
    4. retrieve_word_tfidf_candidates operates consistently on local subsets.
    """
    print("[Verification] Running synthetic verification of consistent cross-chunk scoring...")
    chunk1_texts = ["Acme Logistics International", "Banana Republic Store"]
    chunk2_texts = ["Acme Logistics Corp", "Acme Logistics International", "Delta Airlines"]

    # Pass 1: Build consistent vocabulary & document frequency
    df_counts: Dict[str, int] = {}
    for chunk in [chunk1_texts, chunk2_texts]:
        cv = CountVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            binary=True,
        )
        X = cv.fit_transform(chunk)
        terms = cv.get_feature_names_out()
        c = X.sum(axis=0).A1
        for t, cnt in zip(terms, c):
            df_counts[t] = df_counts.get(t, 0) + int(cnt)

    vocab = {t: i for i, t in enumerate(df_counts.keys())}
    df_arr = np.array(list(df_counts.values()), dtype=np.float32)
    N = len(chunk1_texts) + len(chunk2_texts)
    idf = np.log((N + 1.0) / (df_arr + 1.0)) + 1.0

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        vocabulary=vocab,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vec.idf_ = idf
    vec._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    vec._tfidf.idf_ = idf
    vec._tfidf._idf_diag = sp.diags(
        idf, offsets=0, shape=(len(vocab), len(vocab)), format="csr", dtype=np.float32
    )

    # Query
    q = vec.transform(["Acme Logistics International"])

    # Transform chunks separately using global vectorizer
    X1 = vec.transform(chunk1_texts)
    X2 = vec.transform(chunk2_texts)

    sim1 = q.dot(X1.T).toarray()[0]
    sim2 = q.dot(X2.T).toarray()[0]

    # 'Acme Logistics International' appears in chunk 1 (idx 0) and chunk 2 (idx 1)
    score_chunk1 = sim1[0]
    score_chunk2 = sim2[1]

    if not np.isclose(score_chunk1, score_chunk2, atol=1e-6):
        raise AssertionError(f"Cross-chunk score inconsistency: {score_chunk1} != {score_chunk2}")

    score_partial = sim2[0]  # 'Acme Logistics Corp'
    if not (score_chunk1 > score_partial > 0.0):
        raise AssertionError(f"Ranking inconsistency: exact match {score_chunk1} <= partial match {score_partial}")

    # Verify tie-breaking: identical score items are ordered by entity_id
    items = [("S2-ZED", score_chunk1), ("S2-ABC", score_chunk1)]
    best = heapq.nsmallest(2, items, key=lambda x: (-x[1], x[0]))
    if [cid for cid, _ in best] != ["S2-ABC", "S2-ZED"]:
        raise AssertionError(f"Tie-breaking failure: expected ['S2-ABC', 'S2-ZED'] but got {[cid for cid, _ in best]}")

    # Verify retrieve_word_tfidf_candidates helper on small subset
    s1_mini = pd.DataFrame({"entity_id": ["S1-TEST"], "business_name": ["Acme Logistics International"]})
    tgt_mini = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002"],
        "business_name": ["Acme Logistics International", "Delta Airlines"],
    })
    res_mini = retrieve_word_tfidf_candidates(s1_mini, tgt_mini, top_k=1, return_scores=True)
    if "S2-001" not in res_mini["S1-TEST"]:
        raise AssertionError("retrieve_word_tfidf_candidates failed on miniature test")

    print(f"  [Verification PASSED] Cross-chunk score consistency verified (identical score: {score_chunk1:.6f}).")
    print(f"  [Verification PASSED] Deterministic tie-breaking on entity_id verified.")
    return True


def load_ground_truth_for_s1(
    gt_path: Path,
    s1_id_set: Set[str],
) -> Dict[str, Set[str]]:
    """Stream ground truth and extract matched target IDs for the sampled S1 entities only."""
    gt_map: Dict[str, Set[str]] = {sid: set() for sid in s1_id_set}
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            s1_id = parts[0]
            if s1_id in s1_id_set:
                matched_str = parts[1] if len(parts) > 1 else ""
                if matched_str:
                    matched_ids = set(m.strip() for m in matched_str.split(",") if m.strip())
                    gt_map[s1_id] = matched_ids

    return gt_map


def stream_target_chunks(
    s2_path: Path,
    s3_path: Path,
    chunk_size: int,
    max_target_rows: Optional[int] = None,
) -> Generator[Tuple[pd.DataFrame, str], None, None]:
    """Stream target candidate DataFrames in bounded chunks from Source 2 and Source 3 TSV files.

    Yields:
        Tuple of (target_chunk_df, source_label)
    """
    total_yielded = 0

    for source_label, file_path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not file_path.exists():
            raise FileNotFoundError(f"Target file not found at {file_path}")

        reader = pd.read_csv(
            file_path,
            sep="\t",
            usecols=["entity_id", "business_name"],
            chunksize=chunk_size,
        )

        for chunk_df in reader:
            if max_target_rows is not None and total_yielded >= max_target_rows:
                return

            if max_target_rows is not None and (total_yielded + len(chunk_df)) > max_target_rows:
                chunk_df = chunk_df.iloc[: max_target_rows - total_yielded].copy()

            total_yielded += len(chunk_df)
            yield chunk_df, source_label

            if max_target_rows is not None and total_yielded >= max_target_rows:
                return


def build_consistent_word_tfidf_vectorizer(
    s2_path: Path,
    s3_path: Path,
    chunk_size: int,
    total_target_rows: int,
    max_target_rows: Optional[int] = None,
    ngram_range: Tuple[int, int] = WORD_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Pass 1: Stream target chunks to accumulate global word n-gram document frequencies.

    Builds a single, globally consistent TfidfVectorizer fitted with the exact corpus-wide
    vocabulary and IDF weights without materializing the full 10.3M-row sparse matrix in RAM.
    """
    eval_target_rows = total_target_rows if max_target_rows is None else min(max_target_rows, total_target_rows)
    num_chunks = int(np.ceil(eval_target_rows / chunk_size))

    print(f"\n[Pass 1/2] Streaming {eval_target_rows:,} target records across ~{num_chunks} chunks to build global Word TF-IDF model...")
    t_pass1_start = time.time()
    global_df: Dict[str, int] = {}
    total_processed = 0

    chunk_stream = stream_target_chunks(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=chunk_size,
        max_target_rows=max_target_rows,
    )

    for chunk_idx, (target_chunk_df, source_label) in enumerate(chunk_stream, start=1):
        chunk_len = len(target_chunk_df)
        total_processed += chunk_len

        # Extract normalized names using production normalizer
        names = _extract_normalized_names(target_chunk_df)

        # Count binary document occurrences for word n-grams in this chunk
        cv = CountVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            binary=True,
        )
        X_chunk_counts = cv.fit_transform(names)
        terms = cv.get_feature_names_out()
        counts = X_chunk_counts.sum(axis=0).A1

        # Accumulate into global document frequency dictionary
        for term, cnt in zip(terms, counts):
            global_df[term] = global_df.get(term, 0) + int(cnt)

        del target_chunk_df
        del names
        del cv
        del X_chunk_counts
        gc.collect()

        # Check memory safety during Pass 1
        avail_bytes = psutil.virtual_memory().available
        if avail_bytes < 500 * (1024**2):
            print(f"\n[ERROR] Low memory in Pass 1! Available RAM: {avail_bytes / (1024**2):.1f} MB.")
            sys.exit(3)

        if chunk_idx == 1 or chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            pct = (total_processed / eval_target_rows) * 100.0
            rss_mb = psutil.Process().memory_info().rss / 1e6
            el = time.time() - t_pass1_start
            print(f"  [Pass 1: Chunk {chunk_idx:>3}/{num_chunks}] Processed {total_processed:>10,} targets ({pct:>5.1f}%). Total unique word n-grams: {len(global_df):,} | Elapsed: {el:.1f}s | RSS: {rss_mb:.1f} MB")

    t_pass1_elapsed = time.time() - t_pass1_start
    n_terms = len(global_df)
    print(f"Pass 1 complete in {t_pass1_elapsed:.2f}s. Total global word n-gram vocabulary: {n_terms:,} terms.")

    # Construct global vocabulary and exact IDF array
    print(f"Constructing unified Word TfidfVectorizer (N={total_processed:,}, Vocabulary={n_terms:,})...")
    vocab = {term: idx for idx, term in enumerate(global_df.keys())}
    df_array = np.fromiter(global_df.values(), dtype=np.float32, count=n_terms)
    del global_df
    gc.collect()

    N = float(total_processed)
    # Standard smooth_idf: ln((N + 1) / (DF + 1)) + 1
    idf_vector = np.log((N + 1.0) / (df_array + 1.0)) + 1.0
    del df_array
    gc.collect()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        vocabulary=vocab,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.idf_ = idf_vector
    vectorizer._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    vectorizer._tfidf.idf_ = idf_vector
    vectorizer._tfidf._idf_diag = sp.diags(
        idf_vector, offsets=0, shape=(n_terms, n_terms), format="csr", dtype=np.float32
    )

    print(f"Global Word TfidfVectorizer ready. RSS: {psutil.Process().memory_info().rss / 1e6:.1f} MB")
    return vectorizer


def format_benchmark_results(
    sample_seed: int,
    sample_size: int,
    n_targets: int,
    target_chunk_size: int,
    total_target_chunks: int,
    default_top_k: int,
    global_candidates: Dict[str, Dict[str, float]],
    gt_map: Dict[str, Set[str]],
    top_k_list: List[int],
    runtime_seconds: float,
    peak_memory_mb: float,
) -> Dict:
    """Format and compute multi-top-k blocking evaluation metrics."""
    s1_ids = list(global_candidates.keys())
    n_s1 = len(s1_ids)

    # Compute ground truth total true pairs
    total_true_pairs = sum(len(gt_map.get(sid, set())) for sid in s1_ids)

    top_k_metrics = {}
    for k in sorted(top_k_list):
        retrieved_true_pairs_k = 0
        candidate_counts_k = []

        for sid in s1_ids:
            cand_dict = global_candidates.get(sid, {})
            # Deterministic selection: primary key -score (descending), secondary key cid (ascending)
            items = list(cand_dict.items())
            if len(items) <= k:
                best = sorted(items, key=lambda x: (-x[1], x[0]))
            else:
                best = heapq.nsmallest(k, items, key=lambda x: (-x[1], x[0]))

            top_k_cands = set(cid for cid, _ in best)
            candidate_counts_k.append(len(top_k_cands))

            true_matches = gt_map.get(sid, set())
            retrieved_true_pairs_k += len(true_matches & top_k_cands)

        pair_recall_k = (retrieved_true_pairs_k / total_true_pairs) if total_true_pairs > 0 else 1.0
        avg_cands_k = float(np.mean(candidate_counts_k)) if candidate_counts_k else 0.0
        min_cands_k = int(np.min(candidate_counts_k)) if candidate_counts_k else 0
        max_cands_k = int(np.max(candidate_counts_k)) if candidate_counts_k else 0
        total_cands_k = int(np.sum(candidate_counts_k)) if candidate_counts_k else 0

        top_k_metrics[str(k)] = {
            "top_k": k,
            "pair_recall": round(float(pair_recall_k), 6),
            "retrieved_true_pairs": retrieved_true_pairs_k,
            "total_true_pairs": total_true_pairs,
            "total_retrieved_candidates": total_cands_k,
            "average_candidates_per_s1": round(avg_cands_k, 2),
            "min_candidates_per_s1": min_cands_k,
            "max_candidates_per_s1": max_cands_k,
            "runtime_seconds": round(float(runtime_seconds), 2),
            "peak_memory_mb": round(float(peak_memory_mb), 2),
        }

    # Primary metrics at default_top_k
    primary_k_str = str(default_top_k)
    if primary_k_str not in top_k_metrics:
        primary_k_str = str(sorted(top_k_list)[0])
    primary_metrics = top_k_metrics[primary_k_str]

    results = {
        "benchmark_type": "sampled_training_benchmark",
        "benchmark_channel": "word_tfidf",
        "disclaimer": "This is a sampled training benchmark on S1 queries, not the final full-dataset recall.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_seed": sample_seed,
        "sample_size": n_s1,
        "target_corpus_size": n_targets,
        "target_chunk_size": target_chunk_size,
        "total_target_chunks": total_target_chunks,
        "default_top_k": default_top_k,
        "runtime_seconds": round(float(runtime_seconds), 2),
        "peak_memory_mb": round(float(peak_memory_mb), 2),
        "metrics": {
            "number_of_s1_evaluated": n_s1,
            "total_true_pairs": primary_metrics["total_true_pairs"],
            "retrieved_true_pairs": primary_metrics["retrieved_true_pairs"],
            "total_retrieved_candidates": primary_metrics["total_retrieved_candidates"],
            "pair_recall": primary_metrics["pair_recall"],
            "average_candidates_per_s1": primary_metrics["average_candidates_per_s1"],
            "min_candidates_per_s1": primary_metrics["min_candidates_per_s1"],
            "max_candidates_per_s1": primary_metrics["max_candidates_per_s1"],
        },
        "top_k_comparison": top_k_metrics,
    }

    return results


def run_benchmark(
    sample_size: int = 100,
    seed: int = 42,
    top_k: int = WORD_TFIDF_TOP_K,
    compare_top_k: Optional[List[int]] = None,
    target_chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
    s1_path: Path = TRAIN_SOURCE1,
    s2_path: Path = TRAIN_SOURCE2,
    s3_path: Path = TRAIN_SOURCE3,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    output_path: Path = OUTPUT_DIR / "word_blocking_benchmark.json",
    check_safety_only: bool = False,
) -> Optional[Dict]:
    """Run mathematically valid two-pass word TF-IDF candidate blocking evaluation."""
    top_k_list = compare_top_k or [10, 15, 20, 30, 40]
    if top_k not in top_k_list:
        top_k_list.append(top_k)
    max_k = max(top_k_list)

    print("=" * 78)
    print("WORD TF-IDF BLOCKING BENCHMARK (CONSISTENT GLOBAL MODEL STREAMING)")
    print("=" * 78)
    print(f"Sample size: {sample_size:,} S1 entities (seed: {seed})")
    print(f"Primary Top-K: {top_k} (Evaluating Top-K range: {sorted(top_k_list)})")
    print(f"Target chunk size: {target_chunk_size:,} records")

    # 1. Safety assessment
    s2_lines = sum(1 for _ in open(s2_path, "r", encoding="utf-8")) - 1
    s3_lines = sum(1 for _ in open(s3_path, "r", encoding="utf-8")) - 1
    total_target_rows = s2_lines + s3_lines
    eval_target_rows = total_target_rows if max_target_rows is None else min(max_target_rows, total_target_rows)
    num_chunks = int(np.ceil(eval_target_rows / target_chunk_size))
    print(f"Target corpus: {eval_target_rows:,} rows across ~{num_chunks} chunks.")

    is_safe, req_gb, avail_gb, safety_msg = estimate_computation_safety(target_chunk_size)
    print(safety_msg)

    if check_safety_only:
        return None

    if not is_safe:
        print("\n" + "!" * 78)
        print("STOPPING EXECUTION DUE TO COMPUTATIONAL RESOURCE CONSTRAINT:")
        print(f"Chunk size {target_chunk_size:,} requires ~{req_gb:.2f} GB RAM.")
        print(f"Available system RAM is {avail_gb:.2f} GB.")
        print("Reduce --target-chunk-size to run safely on this machine.")
        print("!" * 78 + "\n")
        sys.exit(2)

    # 2. Run synthetic verification
    verify_consistent_scoring_and_merging()

    t_start = time.time()

    # 3. Load sampled Source 1 entities and ground truth
    print(f"\nLoading and sampling {sample_size:,} S1 entities from {s1_path.name}...")
    s1_full_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name"])
    s1_sample_df = s1_full_df.sample(n=min(sample_size, len(s1_full_df)), random_state=seed).copy()
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_sample_df)

    gt_map = load_ground_truth_for_s1(gt_path, set(s1_ids))
    n_gt_pairs = sum(len(m) for m in gt_map.values())
    n_s1_with_gt = sum(1 for m in gt_map.values() if len(m) > 0)
    print(f"Sampled {len(s1_ids):,} S1 queries. Found {n_gt_pairs:,} true matches across {n_s1_with_gt:,} S1 entities ({len(s1_ids) - n_s1_with_gt:,} singletons).")

    # 4. Pass 1: Build consistent global TfidfVectorizer
    vectorizer = build_consistent_word_tfidf_vectorizer(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=target_chunk_size,
        total_target_rows=total_target_rows,
        max_target_rows=max_target_rows,
        ngram_range=WORD_NGRAM_RANGE,
    )

    # 5. Transform S1 queries ONCE using the global vectorizer
    print(f"\nTransforming S1 queries into global feature space...")
    X_s1 = vectorizer.transform(s1_names)
    if not isinstance(X_s1, sp.csr_matrix):
        X_s1 = X_s1.tocsr()
    print(f"X_s1 shape: {X_s1.shape}, nnz: {X_s1.nnz:,}. Features match global vocabulary exactly.")

    # 6. Pass 2: Stream target candidate chunks and merge global Top-K
    print(f"\n[Pass 2/2] Streaming {eval_target_rows:,} target records across ~{num_chunks} chunks for candidate retrieval...")
    global_candidates: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    total_processed_targets = 0
    t_pass2_start = time.time()

    chunk_stream = stream_target_chunks(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=target_chunk_size,
        max_target_rows=max_target_rows,
    )

    for chunk_idx, (target_chunk_df, source_label) in enumerate(chunk_stream, start=1):
        t_chunk_start = time.time()
        chunk_len = len(target_chunk_df)
        total_processed_targets += chunk_len

        # Check memory safety during execution
        avail_bytes = psutil.virtual_memory().available
        if avail_bytes < 500 * (1024**2):
            print(f"\n[UNSAFE MEMORY CONSTRAINT] Free RAM dropped to {avail_bytes / (1024**2):.1f} MB during Pass 2, chunk {chunk_idx}.")
            print("Stopping execution to prevent system kernel crash.")
            sys.exit(3)

        target_ids = target_chunk_df["entity_id"].astype(str).tolist()
        target_names = _extract_normalized_names(target_chunk_df)

        # Transform target chunk into the EXACT SAME feature space
        X_chunk = vectorizer.transform(target_names)
        if not isinstance(X_chunk, sp.csr_matrix):
            X_chunk = X_chunk.tocsr()
        X_chunk_T = X_chunk.T.tocsr()

        # Row-by-row candidate retrieval against chunk
        for i, sid in enumerate(s1_ids):
            q_vec = X_s1[i : i + 1]
            if q_vec.nnz == 0:
                continue

            row_sims = q_vec.dot(X_chunk_T)
            if not isinstance(row_sims, sp.csr_matrix):
                row_sims = row_sims.tocsr()
            if row_sims.nnz == 0:
                continue

            cols = row_sims.indices
            data = row_sims.data

            s1_map = global_candidates[sid]
            for col, score in zip(cols, data):
                cid = target_ids[col]
                score_val = float(score)
                if cid not in s1_map or score_val > s1_map[cid]:
                    s1_map[cid] = score_val

            # Prune to bound memory: retain current top max_k * 4 highest scoring candidates
            if len(s1_map) > max_k * 4:
                top_items = heapq.nsmallest(
                    max_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                )
                global_candidates[sid] = dict(top_items)

        del target_chunk_df
        del target_ids
        del target_names
        del X_chunk
        del X_chunk_T
        gc.collect()

        t_chunk_elapsed = time.time() - t_chunk_start
        total_elapsed = time.time() - t_pass2_start

        if chunk_idx == 1 or chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            pct = (total_processed_targets / eval_target_rows) * 100.0
            rss_mb = psutil.Process().memory_info().rss / 1e6
            print(f"  [Pass 2: Chunk {chunk_idx:>3}/{num_chunks}] Processed {total_processed_targets:>10,} / {eval_target_rows:,} targets ({pct:>5.1f}%). Chunk time: {t_chunk_elapsed:.2f}s | Elapsed: {total_elapsed:.1f}s | RSS: {rss_mb:.1f} MB")

    total_runtime = time.time() - t_start
    peak_rss = get_peak_memory_mb()

    # Final pruning: ensure each S1 entity has strictly its top max_k global candidates with deterministic tie-breaking
    for sid in s1_ids:
        top_items = heapq.nsmallest(
            max_k, global_candidates[sid].items(), key=lambda x: (-x[1], x[0])
        )
        global_candidates[sid] = dict(top_items)

    print(f"\nPass 2 complete in {time.time() - t_pass2_start:.2f}s (Total benchmark runtime: {total_runtime:.2f}s). Peak RSS: {peak_rss:.1f} MB")

    # 7. Format and save results
    results = format_benchmark_results(
        sample_seed=seed,
        sample_size=len(s1_ids),
        n_targets=total_processed_targets,
        target_chunk_size=target_chunk_size,
        total_target_chunks=num_chunks,
        default_top_k=top_k,
        global_candidates=global_candidates,
        gt_map=gt_map,
        top_k_list=top_k_list,
        runtime_seconds=total_runtime,
        peak_memory_mb=peak_rss,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {output_path}")

    # Print summary table
    print("\n" + "=" * 88)
    print("WORD TF-IDF BLOCKING BENCHMARK (CONSISTENT GLOBAL MODEL)")
    print(f"(S1 Queries: {len(s1_ids):,}, Full Target Corpus: {total_processed_targets:,} rows)")
    print("=" * 88)
    print(f"{'Top-K':<8} {'Pair Recall':<14} {'Retrieved Pairs':<18} {'Total Cands':<14} {'Avg Cands/S1':<14} {'Peak RSS (MB)':<14}")
    print("-" * 88)
    for k_str, m in results["top_k_comparison"].items():
        print(f"{m['top_k']:<8} {m['pair_recall']:<14.4f} {m['retrieved_true_pairs']}/{m['total_true_pairs']:<15} {m['total_retrieved_candidates']:<14} {m['average_candidates_per_s1']:<14.2f} {m['peak_memory_mb']:<14.2f}")
    print("-" * 88)
    print(f"Runtime: {total_runtime:.2f}s ({total_runtime / 60.0:.2f} min) | Peak RSS: {peak_rss:.1f} MB")
    print("=" * 88)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word TF-IDF candidate blocking over full target corpus.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of S1 entities to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--top-k", type=int, default=WORD_TFIDF_TOP_K, help=f"Primary Top-K cutoff (default: {WORD_TFIDF_TOP_K})")
    parser.add_argument("--compare-top-k", nargs="+", type=int, default=[10, 15, 20, 30, 40], help="Top-K values to evaluate")
    parser.add_argument("--target-chunk-size", type=int, default=100000, help="Target chunk size for streaming retrieval (default: 100000)")
    parser.add_argument("--max-target-rows", type=int, default=None, help="Maximum target rows to load (default: None for full corpus)")
    parser.add_argument("--s1-path", type=Path, default=TRAIN_SOURCE1, help="Path to train_source1.tsv")
    parser.add_argument("--s2-path", type=Path, default=TRAIN_SOURCE2, help="Path to train_source2.tsv")
    parser.add_argument("--s3-path", type=Path, default=TRAIN_SOURCE3, help="Path to train_source3.tsv")
    parser.add_argument("--gt-path", type=Path, default=TRAIN_GROUND_TRUTH, help="Path to train_ground_truth.tsv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "word_blocking_benchmark.json", help="Output JSON path")
    parser.add_argument("--check-safety-only", action="store_true", help="Perform computational safety estimation and exit")

    args = parser.parse_args()

    run_benchmark(
        sample_size=args.sample_size,
        seed=args.seed,
        top_k=args.top_k,
        compare_top_k=args.compare_top_k,
        target_chunk_size=args.target_chunk_size,
        max_target_rows=args.max_target_rows,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        gt_path=args.gt_path,
        output_path=args.output,
        check_safety_only=args.check_safety_only,
    )


if __name__ == "__main__":
    main()
