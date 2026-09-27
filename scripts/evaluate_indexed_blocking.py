"""Benchmark inverted-index candidate retrieval on 100 S1 training entities (K=40)."""

import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_union_blocking import load_ground_truth_for_s1
from src.blocking import _extract_normalized_names
from src.config import (
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.inverted_index import (
    build_or_load_country_inverted_index,
    get_peak_memory_mb,
    normalize_country,
    retrieve_candidates_from_index,
)


def run_benchmark():
    t_start = time.time()
    sample_size = 100
    seed = 42
    top_k = 40

    print("=" * 88)
    print("BENCHMARK: INVERTED INDEX CANDIDATE RETRIEVAL (100 S1, K=40)")
    print("=" * 88)

    # 1. Load S1 sample
    s1_full_df = pd.read_csv(TRAIN_SOURCE1, sep="\t", usecols=["entity_id", "business_name", "country"])
    s1_sample_df = s1_full_df.sample(n=sample_size, random_state=seed).copy()
    s1_ids = s1_sample_df["entity_id"].astype(str).tolist()

    gt_map = load_ground_truth_for_s1(TRAIN_GROUND_TRUTH, set(s1_ids))
    total_true_pairs = sum(len(m) for m in gt_map.values())
    print(f"Sampled {len(s1_ids)} S1 queries with {total_true_pairs} true pairs.")

    # 2. Load vectorizers
    cache_dir = OUTPUT_DIR / "cache"
    char_vec = joblib.load(cache_dir / "char_vectorizer.joblib")
    word_vec = joblib.load(cache_dir / "word_vectorizer.joblib")

    # 3. Build or load target inverted indexes
    indexes = build_or_load_country_inverted_index(
        split_name="train",
        s2_path=TRAIN_SOURCE2,
        s3_path=TRAIN_SOURCE3,
        char_vec=char_vec,
        word_vec=word_vec,
        cache_dir=cache_dir,
        chunk_size=500000,
    )

    # 4. Retrieve candidates using inverted index
    t_query_start = time.time()
    cand_map, char_scores, word_scores = retrieve_candidates_from_index(
        s1_df=s1_sample_df,
        country_indexes=indexes,
        char_vec=char_vec,
        word_vec=word_vec,
        char_top_k=top_k,
        word_top_k=top_k,
    )
    t_query_elapsed = time.time() - t_query_start
    print(f"Query retrieval completed in: {t_query_elapsed:.4f}s ({len(s1_ids)/t_query_elapsed:.1f} queries/s)")

    # 5. Evaluate recall against ground truth
    char_ret_pairs = 0
    word_ret_pairs = 0
    union_ret_pairs = 0
    total_union_cands = 0

    for sid in s1_ids:
        c_set = set(char_scores.get(sid, {}).keys())
        w_set = set(word_scores.get(sid, {}).keys())
        u_set = set(cand_map.get(sid, []))

        gt = gt_map.get(sid, set())
        char_ret_pairs += len(c_set & gt)
        word_ret_pairs += len(w_set & gt)
        union_ret_pairs += len(u_set & gt)
        total_union_cands += len(u_set)

    total_runtime = time.time() - t_start
    peak_rss = get_peak_memory_mb()

    recall = union_ret_pairs / total_true_pairs
    char_recall = char_ret_pairs / total_true_pairs
    word_recall = word_ret_pairs / total_true_pairs
    avg_cands = total_union_cands / len(s1_ids)

    print("\n" + "=" * 88)
    print("INVERTED INDEX TRAINING BENCHMARK RESULTS (K=40):")
    print("=" * 88)
    print(f"UNION Pair Recall:              {recall*100:.2f}% ({recall:.6f}) [{union_ret_pairs}/{total_true_pairs} true pairs]")
    print(f"Character Pair Recall:          {char_recall*100:.2f}% ({char_recall:.6f}) [{char_ret_pairs}/{total_true_pairs} true pairs]")
    print(f"Word Pair Recall:               {word_recall*100:.2f}% ({word_recall:.6f}) [{word_ret_pairs}/{total_true_pairs} true pairs]")
    print(f"Total Candidate Pairs:          {total_union_cands:,}")
    print(f"Average Candidates per S1:      {avg_cands:.2f}")
    print(f"Total Runtime:                  {total_runtime:.2f}s ({total_runtime / 60:.2f} min)")
    print(f"Pure Query Runtime:             {t_query_elapsed:.4f}s")
    print(f"Peak Memory (RSS):              {peak_rss:.2f} MB")
    print("=" * 88)

    benchmark_results = {
        "benchmark_name": "Inverted Index Country-Partitioned K=40",
        "sample_size": sample_size,
        "seed": seed,
        "char_top_k": top_k,
        "word_top_k": top_k,
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": union_ret_pairs,
        "pair_recall": float(round(recall, 6)),
        "char_retrieved_true_pairs": char_ret_pairs,
        "char_pair_recall": float(round(char_recall, 6)),
        "word_retrieved_true_pairs": word_ret_pairs,
        "word_pair_recall": float(round(word_recall, 6)),
        "total_candidate_pairs": total_union_cands,
        "avg_candidates_per_s1": float(round(avg_cands, 2)),
        "runtime_seconds": float(round(total_runtime, 2)),
        "query_runtime_seconds": float(round(t_query_elapsed, 4)),
        "peak_memory_mb": float(round(peak_rss, 2)),
    }

    out_path = OUTPUT_DIR / "indexed_blocking_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Saved benchmark results to {out_path}")

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()
