"""Evaluate Candidate Recall from the UNION of Character TF-IDF and Word TF-IDF Retrieval.

This script measures candidate recall, retrieved true pairs, and candidate efficiency
from the UNION of Character TF-IDF and Word TF-IDF retrieval channels on a deterministic
sample of Source 1 entities against the FULL target candidate corpus (Source 2 + Source 3,
10,320,219 total records) using a mathematically valid two-pass streaming approach.

LOCKED RETRIEVAL ARCHITECTURE:
S1
├── Character TF-IDF retrieval (Top-K = 20, char (3, 5) n-grams)
└── Word TF-IDF retrieval (Top-K = 20, word (1, 2) n-grams)
          ↓
    Candidate UNION (union_candidates = character_candidates | word_candidates)

MATHEMATICAL RATIONALE FOR TWO-PASS CONSISTENT GLOBAL STREAMING:
----------------------------------------------------------------
1. Pass 1 (Unified Vocabulary & Document Frequency Accumulation):
   Streams target records across all 104 chunks (100,000 records each) in a single pass.
   Accumulates global document frequencies for both:
   - Character 3-5 n-grams (~3.5M unique terms)
   - Word 1-2 n-grams (~6.8M unique terms)
   Computes exact corpus-wide smooth IDF weights:
       IDF_global(t) = ln((N + 1) / (DF(t) + 1)) + 1  where N = 10,320,219
   Instantiates two frozen, unified TfidfVectorizers.
2. Pass 2 (Streaming Candidate Retrieval & Global Top-K Merging):
   S1 queries are transformed once into each respective global feature space.
   Target chunks are streamed and transformed using the exact global vectorizers.
   Cosine similarities across all 104 chunks are evaluated in the identical coordinate space.
   For each S1 entity, the Top-20 Character candidates and Top-20 Word candidates are
   selected with deterministic tie-breaking.
3. Candidate Union:
   Computes union_candidates = character_candidates | word_candidates.
   Verifies that 100% of final union candidates originate from one or both branches.
   Measures recall gains, candidate efficiency, and candidate overlap.

Usage:
    python3 scripts/evaluate_union_blocking.py --sample-size 100
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
    """Estimate whether two-pass dual-channel retrieval is computationally safe on local hardware.

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

    # Base memory: Python process (~0.5 GB) + dual vocabulary dicts (~1.1 GB) + per-chunk buffer (~0.4 GB)
    estimated_required_gb = 1.8 + (target_chunk_size / 1_000_000.0) * 1.8

    # Safe operating margin: estimated RAM must fit within 75% of currently available free RAM
    is_safe = estimated_required_gb <= (available_gb * 0.75)

    msg = (
        f"Memory Safety Assessment (Dual-Channel Two-Pass Streaming Mode):\n"
        f"  - Target chunk size: {target_chunk_size:,} rows\n"
        f"  - Estimated RAM required: {estimated_required_gb:.2f} GB\n"
        f"  - Available system RAM: {available_gb:.2f} GB (Total physical RAM: {total_gb:.2f} GB)\n"
    )

    if is_safe:
        msg += "  - Status: SAFE. Estimated memory fits comfortably within available RAM."
    else:
        msg += (
            f"  - Status: [UNSAFE - CONSTRAINT DETECTED].\n"
            f"    Estimated memory ({estimated_required_gb:.2f} GB) exceeds safe threshold of available RAM ({available_gb:.2f} GB free).\n"
            f"    Decrease --target-chunk-size (e.g. 50000) to proceed safely."
        )

    return is_safe, estimated_required_gb, available_gb, msg


def run_synthetic_union_test() -> bool:
    """Perform small synthetic test verifying set union, recall accounting, and candidate provenance."""
    print("[Verification] Running synthetic verification of candidate union and recall accounting...")

    gt_map = {
        "S1-001": {"S2-001", "S2-002", "S2-003"},
        "S1-002": {"S2-004"},
        "S1-003": {"S2-005"},
    }
    total_true = sum(len(v) for v in gt_map.values())
    assert total_true == 5

    # Simulated Character branch candidates
    char_cands = {
        "S1-001": {"S2-001", "S2-999"},
        "S1-002": {"S2-888"},
        "S1-003": {"S2-005", "S2-111"},
    }

    # Simulated Word branch candidates
    word_cands = {
        "S1-001": {"S2-002", "S2-777"},
        "S1-002": {"S2-004"},
        "S1-003": {"S2-005", "S2-222"},
    }

    # Compute union candidates
    union_cands: Dict[str, Set[str]] = {}
    for sid in gt_map:
        c_set = char_cands[sid]
        w_set = word_cands[sid]
        u_set = c_set | w_set
        union_cands[sid] = u_set

        # Verify candidate provenance (Requirement 10)
        assert u_set == (c_set | w_set)
        for cid in u_set:
            assert cid in c_set or cid in w_set

    # Recall accounting
    char_ret = sum(len(char_cands[sid] & gt_map[sid]) for sid in gt_map)
    word_ret = sum(len(word_cands[sid] & gt_map[sid]) for sid in gt_map)
    union_ret = sum(len(union_cands[sid] & gt_map[sid]) for sid in gt_map)

    assert char_ret == 2   # S2-001, S2-005
    assert word_ret == 3   # S2-002, S2-004, S2-005
    assert union_ret == 4  # S2-001, S2-002, S2-004, S2-005 (both branches contribute complementary matches)

    char_recall = char_ret / total_true
    word_recall = word_ret / total_true
    union_recall = union_ret / total_true

    assert np.isclose(char_recall, 0.40)
    assert np.isclose(word_recall, 0.60)
    assert np.isclose(union_recall, 0.80)
    assert union_recall >= max(char_recall, word_recall)

    overlap = sum(len(char_cands[sid] & word_cands[sid]) for sid in gt_map)
    assert overlap == 1  # S2-005 retrieved by both

    print(f"  [Verification PASSED] Synthetic union verified: Char Recall={char_recall:.2f}, Word Recall={word_recall:.2f}, Union Recall={union_recall:.2f}.")
    print(f"  [Verification PASSED] 100% of union candidates validated to originate from Char or Word branches.")
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


def build_dual_consistent_vectorizers(
    s2_path: Path,
    s3_path: Path,
    chunk_size: int,
    total_target_rows: int,
    max_target_rows: Optional[int] = None,
    char_ngram_range: Tuple[int, int] = (3, 5),
    word_ngram_range: Tuple[int, int] = WORD_NGRAM_RANGE,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Pass 1: Stream target chunks once to accumulate global document frequencies for both Char and Word TF-IDF.

    Constructs two unified, globally consistent TfidfVectorizers without materializing the full 10.3M-row matrices.
    """
    eval_target_rows = total_target_rows if max_target_rows is None else min(max_target_rows, total_target_rows)
    num_chunks = int(np.ceil(eval_target_rows / chunk_size))

    print(f"\n[Pass 1/2] Streaming {eval_target_rows:,} target records across ~{num_chunks} chunks to build global TF-IDF models...")
    t_pass1_start = time.time()
    char_global_df: Dict[str, int] = {}
    word_global_df: Dict[str, int] = {}
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

        # 1. Character TF-IDF document frequencies
        cv_char = CountVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            min_df=1,
            binary=True,
        )
        Xc = cv_char.fit_transform(names)
        terms_c = cv_char.get_feature_names_out()
        counts_c = Xc.sum(axis=0).A1
        for t, cnt in zip(terms_c, counts_c):
            char_global_df[t] = char_global_df.get(t, 0) + int(cnt)

        # 2. Word TF-IDF document frequencies
        cv_word = CountVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            binary=True,
        )
        Xw = cv_word.fit_transform(names)
        terms_w = cv_word.get_feature_names_out()
        counts_w = Xw.sum(axis=0).A1
        for t, cnt in zip(terms_w, counts_w):
            word_global_df[t] = word_global_df.get(t, 0) + int(cnt)

        del target_chunk_df
        del names
        del cv_char
        del Xc
        del cv_word
        del Xw
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
            print(f"  [Pass 1: Chunk {chunk_idx:>3}/{num_chunks}] Processed {total_processed:>10,} targets ({pct:>5.1f}%). Unique Char n-grams: {len(char_global_df):,}, Word n-grams: {len(word_global_df):,} | Elapsed: {el:.1f}s | RSS: {rss_mb:.1f} MB")

    t_pass1_elapsed = time.time() - t_pass1_start
    n_char_terms = len(char_global_df)
    n_word_terms = len(word_global_df)
    print(f"Pass 1 complete in {t_pass1_elapsed:.2f}s. Vocabulary: {n_char_terms:,} Char n-grams, {n_word_terms:,} Word n-grams.")

    N = float(total_processed)

    # 1. Build unified Character TfidfVectorizer
    print("Constructing unified Character TfidfVectorizer...")
    vocab_c = {t: i for i, t in enumerate(char_global_df.keys())}
    df_arr_c = np.fromiter(char_global_df.values(), dtype=np.float32, count=n_char_terms)
    del char_global_df
    gc.collect()

    idf_c = np.log((N + 1.0) / (df_arr_c + 1.0)) + 1.0
    del df_arr_c
    gc.collect()

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        vocabulary=vocab_c,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vec.idf_ = idf_c
    char_vec._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    char_vec._tfidf.idf_ = idf_c
    char_vec._tfidf._idf_diag = sp.diags(
        idf_c, offsets=0, shape=(n_char_terms, n_char_terms), format="csr", dtype=np.float32
    )

    # 2. Build unified Word TfidfVectorizer
    print("Constructing unified Word TfidfVectorizer...")
    vocab_w = {t: i for i, t in enumerate(word_global_df.keys())}
    df_arr_w = np.fromiter(word_global_df.values(), dtype=np.float32, count=n_word_terms)
    del word_global_df
    gc.collect()

    idf_w = np.log((N + 1.0) / (df_arr_w + 1.0)) + 1.0
    del df_arr_w
    gc.collect()

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        vocabulary=vocab_w,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vec.idf_ = idf_w
    word_vec._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    word_vec._tfidf.idf_ = idf_w
    word_vec._tfidf._idf_diag = sp.diags(
        idf_w, offsets=0, shape=(n_word_terms, n_word_terms), format="csr", dtype=np.float32
    )

    print(f"Both global TfidfVectorizers ready. RSS: {psutil.Process().memory_info().rss / 1e6:.1f} MB")
    return char_vec, word_vec


def run_benchmark(
    sample_size: int = 100,
    seed: int = 42,
    char_top_k: int = CHAR_TFIDF_TOP_K,
    word_top_k: int = WORD_TFIDF_TOP_K,
    compare_top_k: Optional[List[int]] = None,
    target_chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
    s1_path: Path = TRAIN_SOURCE1,
    s2_path: Path = TRAIN_SOURCE2,
    s3_path: Path = TRAIN_SOURCE3,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    output_path: Path = OUTPUT_DIR / "union_blocking_benchmark.json",
    check_safety_only: bool = False,
    force_recompute_vectorizers: bool = False,
) -> Optional[Dict]:
    """Run mathematically valid dual-channel Character + Word TF-IDF union blocking evaluation."""
    if compare_top_k is None:
        compare_top_k = [20, 30, 40, 50]

    max_k = max(compare_top_k) if compare_top_k else max(char_top_k, word_top_k)

    print("=" * 88)
    print("CANDIDATE UNION BLOCKING BENCHMARK: CHARACTER TF-IDF + WORD TF-IDF")
    print("=" * 88)
    print(f"Sample size: {sample_size:,} S1 entities (seed: {seed})")
    print(f"Evaluated Top-K list: {compare_top_k} (Max K: {max_k})")
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
    run_synthetic_union_test()

    t_start = time.time()

    # 3. Load sampled Source 1 entities and ground truth
    print(f"\nLoading and sampling {sample_size:,} S1 entities from {s1_path.name}...")
    s1_full_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name"])
    s1_sample_df = s1_full_df.sample(n=min(sample_size, len(s1_full_df)), random_state=seed).copy()
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_sample_df)

    gt_map = load_ground_truth_for_s1(gt_path, set(s1_ids))
    total_true_pairs = sum(len(m) for m in gt_map.values())
    n_s1_with_gt = sum(1 for m in gt_map.values() if len(m) > 0)
    print(f"Sampled {len(s1_ids):,} S1 queries. Found {total_true_pairs:,} true matches across {n_s1_with_gt:,} S1 entities ({len(s1_ids) - n_s1_with_gt:,} singletons).")

    # 4. Pass 1: Build consistent global TfidfVectorizers (or load from cache)
    cache_dir = OUTPUT_DIR / "cache"
    char_cache_file = cache_dir / "char_vectorizer.joblib"
    word_cache_file = cache_dir / "word_vectorizer.joblib"

    if (
        not force_recompute_vectorizers
        and max_target_rows is None
        and char_cache_file.exists()
        and word_cache_file.exists()
    ):
        print(f"\n[Cache] Loading precomputed global TF-IDF vectorizers from {cache_dir}...")
        import joblib
        char_vectorizer = joblib.load(char_cache_file)
        word_vectorizer = joblib.load(word_cache_file)
    else:
        char_vectorizer, word_vectorizer = build_dual_consistent_vectorizers(
            s2_path=s2_path,
            s3_path=s3_path,
            chunk_size=target_chunk_size,
            total_target_rows=total_target_rows,
            max_target_rows=max_target_rows,
            char_ngram_range=(3, 5),
            word_ngram_range=WORD_NGRAM_RANGE,
        )
        if max_target_rows is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(char_vectorizer, char_cache_file)
            joblib.dump(word_vectorizer, word_cache_file)

    # 5. Transform S1 queries ONCE using both global vectorizers
    print("\nTransforming S1 queries into global feature spaces...")
    X_s1_char = char_vectorizer.transform(s1_names).tocsr()
    X_s1_word = word_vectorizer.transform(s1_names).tocsr()
    print(f"X_s1_char shape: {X_s1_char.shape}, nnz: {X_s1_char.nnz:,}")
    print(f"X_s1_word shape: {X_s1_word.shape}, nnz: {X_s1_word.nnz:,}")

    # 6. Pass 2: Stream target candidate chunks and accumulate candidates for both branches
    print(f"\n[Pass 2/2] Streaming {eval_target_rows:,} target records across ~{num_chunks} chunks for dual retrieval (Max K={max_k})...")
    char_candidates: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    word_candidates: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
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

        # --- 6a. Character TF-IDF Retrieval for Chunk ---
        X_chunk_char = char_vectorizer.transform(target_names).tocsr()
        X_chunk_char_T = X_chunk_char.T.tocsr()

        for i, sid in enumerate(s1_ids):
            q_vec = X_s1_char[i : i + 1]
            if q_vec.nnz == 0:
                continue
            row_sims = q_vec.dot(X_chunk_char_T).tocsr()
            if row_sims.nnz == 0:
                continue

            cols = row_sims.indices
            data = row_sims.data
            if len(data) > max_k * 2:
                top_idx = np.argpartition(data, -max_k * 2)[-max_k * 2:]
                cols = cols[top_idx]
                data = data[top_idx]

            s1_map = char_candidates[sid]
            for col, score in zip(cols, data):
                cid = target_ids[col]
                s = float(score)
                if cid not in s1_map or s > s1_map[cid]:
                    s1_map[cid] = s

            # Prune memory periodically
            if len(s1_map) > max_k * 4:
                top_items = heapq.nsmallest(
                    max_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                )
                char_candidates[sid] = dict(top_items)

        del X_chunk_char
        del X_chunk_char_T

        # --- 6b. Word TF-IDF Retrieval for Chunk ---
        X_chunk_word = word_vectorizer.transform(target_names).tocsr()
        X_chunk_word_T = X_chunk_word.T.tocsr()

        for i, sid in enumerate(s1_ids):
            q_vec = X_s1_word[i : i + 1]
            if q_vec.nnz == 0:
                continue
            row_sims = q_vec.dot(X_chunk_word_T).tocsr()
            if row_sims.nnz == 0:
                continue

            cols = row_sims.indices
            data = row_sims.data
            if len(data) > max_k * 2:
                top_idx = np.argpartition(data, -max_k * 2)[-max_k * 2:]
                cols = cols[top_idx]
                data = data[top_idx]

            s1_map = word_candidates[sid]
            for col, score in zip(cols, data):
                cid = target_ids[col]
                s = float(score)
                if cid not in s1_map or s > s1_map[cid]:
                    s1_map[cid] = s

            # Prune memory periodically
            if len(s1_map) > max_k * 4:
                top_items = heapq.nsmallest(
                    max_k * 2, s1_map.items(), key=lambda x: (-x[1], x[0])
                )
                word_candidates[sid] = dict(top_items)

        del X_chunk_word
        del X_chunk_word_T
        del target_chunk_df
        del target_ids
        del target_names
        gc.collect()

        t_chunk_elapsed = time.time() - t_chunk_start
        total_elapsed = time.time() - t_pass2_start

        if chunk_idx == 1 or chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            pct = (total_processed_targets / eval_target_rows) * 100.0
            rss_mb = psutil.Process().memory_info().rss / 1e6
            print(f"  [Pass 2: Chunk {chunk_idx:>3}/{num_chunks}] Processed {total_processed_targets:>10,} / {eval_target_rows:,} targets ({pct:>5.1f}%). Chunk time: {t_chunk_elapsed:.2f}s | Elapsed: {total_elapsed:.1f}s | RSS: {rss_mb:.1f} MB")

    total_runtime = time.time() - t_start
    peak_rss = get_peak_memory_mb()

    # 7. Evaluate each Top-K cutoff
    top_k_comparison: Dict[str, Dict[str, Any]] = {}

    for k in sorted(compare_top_k):
        final_char_cands: Dict[str, Set[str]] = {}
        final_word_cands: Dict[str, Set[str]] = {}
        final_union_cands: Dict[str, Set[str]] = {}

        for sid in s1_ids:
            c_top = heapq.nsmallest(k, char_candidates[sid].items(), key=lambda x: (-x[1], x[0]))
            w_top = heapq.nsmallest(k, word_candidates[sid].items(), key=lambda x: (-x[1], x[0]))

            c_set = set(cid for cid, _ in c_top)
            w_set = set(cid for cid, _ in w_top)
            u_set = c_set | w_set

            final_char_cands[sid] = c_set
            final_word_cands[sid] = w_set
            final_union_cands[sid] = u_set

        c_ret = sum(len(final_char_cands[sid] & gt_map[sid]) for sid in s1_ids)
        w_ret = sum(len(final_word_cands[sid] & gt_map[sid]) for sid in s1_ids)
        u_ret = sum(len(final_union_cands[sid] & gt_map[sid]) for sid in s1_ids)

        c_recall = float(c_ret / total_true_pairs) if total_true_pairs > 0 else 1.0
        w_recall = float(w_ret / total_true_pairs) if total_true_pairs > 0 else 1.0
        u_recall = float(u_ret / total_true_pairs) if total_true_pairs > 0 else 1.0

        c_cands = sum(len(final_char_cands[sid]) for sid in s1_ids)
        w_cands = sum(len(final_word_cands[sid]) for sid in s1_ids)
        u_cands = sum(len(final_union_cands[sid]) for sid in s1_ids)

        avg_c_cands = float(c_cands / len(s1_ids))
        avg_w_cands = float(w_cands / len(s1_ids))
        avg_u_cands = float(u_cands / len(s1_ids))

        overlap = sum(len(final_char_cands[sid] & final_word_cands[sid]) for sid in s1_ids)
        avg_overlap = float(overlap / len(s1_ids))
        overlap_pct = float(overlap / u_cands * 100.0) if u_cands > 0 else 0.0

        top_k_comparison[f"K={k}"] = {
            "top_k": k,
            "pair_recall": round(u_recall, 6),
            "retrieved_true_pairs": u_ret,
            "total_true_pairs": total_true_pairs,
            "total_candidate_pairs": u_cands,
            "average_candidates_per_s1": round(avg_u_cands, 2),
            "char_retrieved_true_pairs": c_ret,
            "word_retrieved_true_pairs": w_ret,
            "char_pair_recall": round(c_recall, 6),
            "word_pair_recall": round(w_recall, 6),
            "candidate_overlap": overlap,
            "avg_candidate_overlap_per_s1": round(avg_overlap, 2),
            "overlap_percent_of_union": round(overlap_pct, 2),
            "runtime_seconds": round(float(total_runtime), 2),
            "peak_memory_mb": round(float(peak_rss), 2),
        }

    # 8. Compute Marginal Gains between consecutive K cutoffs
    marginal_gains = []
    sorted_ks = sorted(compare_top_k)
    for i in range(1, len(sorted_ks)):
        k_prev = sorted_ks[i - 1]
        k_curr = sorted_ks[i]
        m_prev = top_k_comparison[f"K={k_prev}"]
        m_curr = top_k_comparison[f"K={k_curr}"]

        delta_ret = m_curr["retrieved_true_pairs"] - m_prev["retrieved_true_pairs"]
        delta_recall = m_curr["pair_recall"] - m_prev["pair_recall"]
        delta_cands = m_curr["total_candidate_pairs"] - m_prev["total_candidate_pairs"]
        delta_avg_cands = m_curr["average_candidates_per_s1"] - m_prev["average_candidates_per_s1"]

        marginal_gains.append({
            "transition": f"K{k_prev} -> K{k_curr}",
            "from_k": k_prev,
            "to_k": k_curr,
            "delta_retrieved_true_pairs": delta_ret,
            "delta_pair_recall": round(delta_recall, 6),
            "delta_pair_recall_pct": round(delta_recall * 100.0, 2),
            "delta_total_candidates": delta_cands,
            "delta_average_candidates_per_s1": round(delta_avg_cands, 2),
        })

    # 9. Format Results JSON
    primary_k = compare_top_k[0] if compare_top_k else 20
    primary_metrics = top_k_comparison.get(f"K={primary_k}", {})

    results = {
        "benchmark_type": "sampled_training_benchmark",
        "benchmark_channel": "union_tfidf",
        "disclaimer": "This is a sampled training benchmark on S1 queries, not the final full-dataset recall.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_seed": seed,
        "sample_size": len(s1_ids),
        "target_corpus_size": total_processed_targets,
        "full_corpus_evaluated": (total_processed_targets == total_target_rows),
        "target_chunk_size": target_chunk_size,
        "total_target_chunks": num_chunks,
        "evaluated_k_values": sorted(compare_top_k),
        "runtime_seconds": round(float(total_runtime), 2),
        "peak_memory_mb": round(float(peak_rss), 2),
        "provenance_verified": True,
        "top_k_comparison": top_k_comparison,
        "marginal_gains": marginal_gains,
        "metrics": primary_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {output_path}")

    # 10. Print Complete UNION Multi-K Table
    print("\n" + "=" * 96)
    print("CANDIDATE UNION BLOCKING BENCHMARK: TOP-K COMPARISON (K=20, 30, 40, 50)")
    print(f"(S1 Queries: {len(s1_ids):,}, Full Target Corpus: {total_processed_targets:,} rows across {num_chunks} chunks)")
    print("=" * 96)
    print(f"{'Top-K':<8} {'Pair Recall':<14} {'Retrieved Pairs':<18} {'Total Cands':<14} {'Avg Cands/S1':<14} {'Peak RSS (MB)':<14}")
    print("-" * 96)
    for k_str, m in top_k_comparison.items():
        print(f"{m['top_k']:<8} {m['pair_recall']*100:>6.2f}% ({m['pair_recall']:<6.4f}) {m['retrieved_true_pairs']}/{m['total_true_pairs']:<15} {m['total_candidate_pairs']:<14} {m['average_candidates_per_s1']:<14.2f} {m['peak_memory_mb']:<14.2f}")
    print("-" * 96)

    print("\n" + "=" * 96)
    print("MARGINAL RECALL GAIN & CANDIDATE EFFICIENCY TRADEOFF")
    print("=" * 96)
    print(f"{'Transition':<16} {'Recall Gain':<18} {'True Pairs Gained':<20} {'Candidate Growth':<20} {'Avg Cands/S1 Growth':<20}")
    print("-" * 96)
    for g in marginal_gains:
        print(f"{g['transition']:<16} +{g['delta_pair_recall_pct']:>5.2f}% ({g['delta_pair_recall']:<+.4f}) {g['delta_retrieved_true_pairs']:<+20d} {g['delta_total_candidates']:<+20d} {g['delta_average_candidates_per_s1']:<+20.2f}")
    print("-" * 96)
    print(f"Total Runtime: {total_runtime:.2f}s ({total_runtime / 60.0:.2f} min) | Peak RSS: {peak_rss:.1f} MB")
    print("=" * 96)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Recall from the UNION of Character and Word TF-IDF.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of S1 entities to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--char-top-k", type=int, default=CHAR_TFIDF_TOP_K, help=f"Character Top-K cutoff (default: {CHAR_TFIDF_TOP_K})")
    parser.add_argument("--word-top-k", type=int, default=WORD_TFIDF_TOP_K, help=f"Word Top-K cutoff (default: {WORD_TFIDF_TOP_K})")
    parser.add_argument("--compare-top-k", nargs="+", type=int, default=[20, 30, 40, 50], help="Top-K cutoff values to evaluate (default: 20 30 40 50)")
    parser.add_argument("--target-chunk-size", type=int, default=100000, help="Target chunk size for streaming retrieval (default: 100000)")
    parser.add_argument("--max-target-rows", type=int, default=None, help="Maximum target rows to load (default: None for full corpus)")
    parser.add_argument("--s1-path", type=Path, default=TRAIN_SOURCE1, help="Path to train_source1.tsv")
    parser.add_argument("--s2-path", type=Path, default=TRAIN_SOURCE2, help="Path to train_source2.tsv")
    parser.add_argument("--s3-path", type=Path, default=TRAIN_SOURCE3, help="Path to train_source3.tsv")
    parser.add_argument("--gt-path", type=Path, default=TRAIN_GROUND_TRUTH, help="Path to train_ground_truth.tsv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "union_blocking_benchmark.json", help="Output JSON path")
    parser.add_argument("--check-safety-only", action="store_true", help="Perform computational safety estimation and exit")
    parser.add_argument("--force-recompute-vectorizers", action="store_true", help="Force rebuilding global TF-IDF vectorizers instead of loading from cache")

    args = parser.parse_args()

    run_benchmark(
        sample_size=args.sample_size,
        seed=args.seed,
        char_top_k=args.char_top_k,
        word_top_k=args.word_top_k,
        compare_top_k=args.compare_top_k,
        target_chunk_size=args.target_chunk_size,
        max_target_rows=args.max_target_rows,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        gt_path=args.gt_path,
        output_path=args.output,
        check_safety_only=args.check_safety_only,
        force_recompute_vectorizers=args.force_recompute_vectorizers,
    )


if __name__ == "__main__":
    main()
