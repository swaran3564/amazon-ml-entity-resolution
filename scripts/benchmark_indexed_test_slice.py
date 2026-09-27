"""Benchmark Inverted Index Candidate Retrieval on 10,000 Test S1 Entities.

Measures:
1. wall-clock runtime
2. peak total aggregate RSS
3. candidate count
4. predicted match count
5. average candidates/S1
6. average matches/S1
7. output sizes
8. official validator result
9. comparison against pilot_2w outputs (equality, duplicates, missing S1)
10. projected full-test runtime
"""

import argparse
import gc
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_test_inference import collect_target_records_for_ids, merge_worker_outputs
from src.config import (
    OUTPUT_DIR,
    TEST_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from src.features import FEATURE_COLUMNS, build_feature_matrix
from src.inverted_index import (
    build_or_load_country_inverted_index,
    get_peak_memory_mb,
    retrieve_candidates_from_index,
)
from utils.validate_submission import validate


def run_benchmark(n_s1: int = 10000):
    t_start = time.time()
    threshold = 0.9000
    top_k = 40

    cand_out = OUTPUT_DIR / "pilot_indexed_candidates.tsv"
    match_out = OUTPUT_DIR / "pilot_indexed_matches.tsv"

    print("=" * 88)
    print("BENCHMARK: INVERTED INDEX CANDIDATE RETRIEVAL ON 10,000 TEST S1")
    print("=" * 88)
    print(f"Test S1 File:           {TEST_SOURCE1}")
    print(f"Test S2 File:           {TEST_SOURCE2}")
    print(f"Test S3 File:           {TEST_SOURCE3}")
    print(f"Model Path:             {OUTPUT_DIR / 'matcher_model_k40.txt'}")
    print(f"Decision Threshold:     {threshold:.4f}")
    print(f"Candidate Cutoff:       Top-{top_k} Char UNION Top-{top_k} Word")
    print(f"Output Candidates:      {cand_out}")
    print(f"Output Matching:        {match_out}")
    print("=" * 88)

    # 1. Load S1 test queries (first 10,000)
    s1_full_df = pd.read_csv(TEST_SOURCE1, sep="\t", nrows=n_s1)
    s1_ids = s1_full_df["entity_id"].astype(str).tolist()
    print(f"Loaded {len(s1_ids):,} test S1 entities.")

    # 2. Load vectorizers and model
    cache_dir = OUTPUT_DIR / "cache"
    char_vec = joblib.load(cache_dir / "char_vectorizer.joblib")
    word_vec = joblib.load(cache_dir / "word_vectorizer.joblib")
    booster = lgb.Booster(model_file=str(OUTPUT_DIR / "matcher_model_k40.txt"))

    # 3. Build or load target inverted index for test split
    t_index_start = time.time()
    test_indexes = build_or_load_country_inverted_index(
        split_name="test",
        s2_path=TEST_SOURCE2,
        s3_path=TEST_SOURCE3,
        char_vec=char_vec,
        word_vec=word_vec,
        cache_dir=cache_dir,
        chunk_size=500000,
    )
    t_index_elapsed = time.time() - t_index_start
    print(f"Index readiness time: {t_index_elapsed:.2f}s | RSS: {get_peak_memory_mb():.1f} MB")

    # 4. Fast inverted index candidate retrieval
    t_query_start = time.time()
    cands_map, char_scores, word_scores = retrieve_candidates_from_index(
        s1_df=s1_full_df,
        country_indexes=test_indexes,
        char_vec=char_vec,
        word_vec=word_vec,
        char_top_k=top_k,
        word_top_k=top_k,
    )
    t_query_elapsed = time.time() - t_query_start
    print(f"Inverted index candidate retrieval: {t_query_elapsed:.2f}s ({n_s1/t_query_elapsed:.1f} S1/sec) | RSS: {get_peak_memory_mb():.1f} MB")

    # 5. Form candidate pairs
    batch_pairs = []
    needed_cand_ids = set()
    for sid, c_list in cands_map.items():
        for cid in c_list:
            batch_pairs.append({
                "source1_entity_id": sid,
                "matched_entity_id": cid,
                "char_score": char_scores[sid].get(cid, 0.0),
                "word_score": word_scores[sid].get(cid, 0.0),
            })
            needed_cand_ids.add(cid)

    print(f"Total candidate pairs to score: {len(batch_pairs):,} across {len(needed_cand_ids):,} distinct targets.")

    # 6. Retrieve target attributes for candidate pairs
    t_target_start = time.time()
    target_dict = collect_target_records_for_ids(needed_cand_ids, TEST_SOURCE2, TEST_SOURCE3, chunk_size=200000)
    t_target_elapsed = time.time() - t_target_start
    print(f"Target record retrieval: {t_target_elapsed:.2f}s | RSS: {get_peak_memory_mb():.1f} MB")

    # 7. Feature engineering & model scoring
    t_score_start = time.time()
    if "business_address" in s1_full_df.columns:
        s1_full_df["address"] = s1_full_df["business_address"]
    s1_lookup = s1_full_df.set_index("entity_id").to_dict(orient="index")

    matched_map: Dict[str, List[str]] = {str(sid): [] for sid in s1_ids}
    if batch_pairs:
        pairs_df = pd.DataFrame(batch_pairs)
        feat_df = build_feature_matrix(
            candidate_pairs=pairs_df[["source1_entity_id", "matched_entity_id", "char_score", "word_score"]],
            s1_lookup=s1_lookup,
            target_lookup=target_dict,
            chunk_size=10000,
        )
        probs = booster.predict(feat_df[FEATURE_COLUMNS].values)
        for (sid, cid), prob in zip(zip(pairs_df["source1_entity_id"], pairs_df["matched_entity_id"]), probs):
            if prob >= threshold:
                matched_map[sid].append((cid, float(prob)))

        for sid in matched_map:
            matched_map[sid] = [cid for cid, _ in sorted(matched_map[sid], key=lambda x: -x[1])]

    t_score_elapsed = time.time() - t_score_start
    print(f"Feature engineering & scoring: {t_score_elapsed:.2f}s | RSS: {get_peak_memory_mb():.1f} MB")

    # 8. Deterministic merge & write final outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        raw_cand_path = tmp_p / "raw_cand.tsv"
        raw_match_path = tmp_p / "raw_match.tsv"

        with open(raw_cand_path, "w", encoding="utf-8") as f_c, open(raw_match_path, "w", encoding="utf-8") as f_m:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in s1_ids:
                f_c.write(f"{sid}\t{','.join(cands_map.get(sid, []))}\n")
                f_m.write(f"{sid}\t{','.join(matched_map.get(sid, []))}\n")

        merge_stats = merge_worker_outputs(
            shard_cand_paths=[raw_cand_path],
            shard_match_paths=[raw_match_path],
            required_s1_ids=s1_ids,
            final_cand_path=cand_out,
            final_match_path=match_out,
        )

    total_runtime = time.time() - t_start
    peak_rss = get_peak_memory_mb()
    cand_size_bytes = cand_out.stat().st_size
    match_size_bytes = match_out.stat().st_size

    # 9. Official validator run
    print("\nRunning official submission validator...")
    with tempfile.TemporaryDirectory() as tmp_vdir:
        tmp_vp = Path(tmp_vdir)
        s1_vpath = tmp_vp / "test_source1.tsv"
        s1_full_df[["entity_id", "business_name", "business_address", "country"]].to_csv(
            s1_vpath, sep="\t", index=False
        )
        # Symlink S2 and S3 for ID validation
        os.symlink(TEST_SOURCE2, tmp_vp / "test_source2.tsv")
        os.symlink(TEST_SOURCE3, tmp_vp / "test_source3.tsv")

        errors, warnings = validate(
            matching_path=str(match_out),
            candidate_path=str(cand_out),
            test_dir=str(tmp_vp),
            check_ids=True,
        )

    val_pass = len(errors) == 0
    print(f"Validator Status: {'PASS' if val_pass else 'FAIL'} (Errors: {len(errors)}, Warnings: {len(warnings)})")

    # 10. Compare against pilot_2w outputs
    p2w_cand = OUTPUT_DIR / "pilot_2w_candidates.tsv"
    p2w_match = OUTPUT_DIR / "pilot_2w_matches.tsv"

    cand_identical = False
    match_identical = False
    cand_overlap = 0.0

    if p2w_cand.exists() and p2w_match.exists():
        p2w_c_df = pd.read_csv(p2w_cand, sep="\t", keep_default_na=False)
        cur_c_df = pd.read_csv(cand_out, sep="\t", keep_default_na=False)
        cand_identical = cur_c_df.equals(p2w_c_df)

        p2w_m_df = pd.read_csv(p2w_match, sep="\t", keep_default_na=False)
        cur_m_df = pd.read_csv(match_out, sep="\t", keep_default_na=False)
        match_identical = cur_m_df.equals(p2w_m_df)

        # Compute candidate set jaccard similarity across S1 entities
        jaccards = []
        for c1, c2 in zip(cur_c_df["candidate_entity_ids"], p2w_c_df["candidate_entity_ids"]):
            s1 = set(c1.split(",")) if c1 else set()
            s2 = set(c2.split(",")) if c2 else set()
            if not s1 and not s2:
                jaccards.append(1.0)
            else:
                jaccards.append(len(s1 & s2) / len(s1 | s2))
        cand_overlap = float(np.mean(jaccards))

    throughput_s1_sec = n_s1 / total_runtime
    projected_full_hours = (1732544 / throughput_s1_sec) / 3600.0

    print("\n" + "=" * 88)
    print("BENCHMARK SUMMARY (10,000 TEST S1 INVERTED INDEX RETRIEVAL):")
    print("=" * 88)
    print(f"Wall-Clock Runtime:                     {total_runtime:.2f}s ({total_runtime / 60:.2f} min)")
    print(f"Throughput:                             {throughput_s1_sec:.2f} S1 / sec")
    print(f"Pure Query Retrieval Time:              {t_query_elapsed:.2f}s ({n_s1/t_query_elapsed:.1f} S1 / sec)")
    print(f"Peak Total Memory (RSS):                {peak_rss:.2f} MB ({peak_rss / 1024:.2f} GB)")
    print(f"Total Candidate IDs:                    {merge_stats['total_candidate_ids']:,}")
    print(f"Average Candidates per S1:              {merge_stats['total_candidate_ids'] / n_s1:.2f}")
    print(f"Total Predicted Matches:                {merge_stats['total_predicted_matches']:,}")
    print(f"Average Predicted Matches per S1:       {merge_stats['total_predicted_matches'] / n_s1:.4f}")
    print(f"Zero-Match S1 Count:                    {merge_stats['zero_match_s1_count']:,}")
    print(f"Zero-Candidate S1 Count:                {merge_stats['zero_candidate_s1_count']:,}")
    print(f"Candidate TSV Size:                     {cand_size_bytes / 1e6:.2f} MB")
    print(f"Matching TSV Size:                      {match_size_bytes / 1e6:.2f} MB")
    print(f"Validator Status:                       {'PASS' if val_pass else 'FAIL'}")
    print(f"Candidate Output Bitwise Identical:     {cand_identical}")
    print(f"Matching Output Bitwise Identical:      {match_identical}")
    print(f"Candidate Jaccard Overlap with Pilot2W: {cand_overlap * 100:.2f}%")
    print(f"Projected Full Test Runtime (1.73M S1): {projected_full_hours:.2f} hours")
    print("=" * 88)

    benchmark_dict = {
        "benchmark_name": "Inverted Index 10k Test Slice",
        "n_s1": n_s1,
        "runtime_seconds": float(round(total_runtime, 2)),
        "throughput_s1_per_sec": float(round(throughput_s1_sec, 2)),
        "pure_query_runtime_seconds": float(round(t_query_elapsed, 2)),
        "peak_rss_mb": float(round(peak_rss, 2)),
        "total_candidate_ids": merge_stats["total_candidate_ids"],
        "avg_candidates_per_s1": float(round(merge_stats["total_candidate_ids"] / n_s1, 2)),
        "total_predicted_matches": merge_stats["total_predicted_matches"],
        "avg_predicted_matches_per_s1": float(round(merge_stats["total_predicted_matches"] / n_s1, 4)),
        "zero_match_s1_count": merge_stats["zero_match_s1_count"],
        "zero_candidate_s1_count": merge_stats["zero_candidate_s1_count"],
        "candidate_tsv_size_bytes": cand_size_bytes,
        "matching_tsv_size_bytes": match_size_bytes,
        "validator_passed": val_pass,
        "cand_identical_to_pilot2w": cand_identical,
        "match_identical_to_pilot2w": match_identical,
        "cand_jaccard_overlap_pct": float(round(cand_overlap * 100, 2)),
        "projected_full_test_hours": float(round(projected_full_hours, 2)),
    }

    out_json = OUTPUT_DIR / "indexed_test_benchmark_10k.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)
    print(f"Saved test benchmark results to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Inverted Index Candidate Retrieval")
    parser.add_argument("--n-s1", type=int, default=10000, help="Number of S1 test entities to evaluate (default: 10000)")
    args = parser.parse_args()
    run_benchmark(n_s1=args.n_s1)
