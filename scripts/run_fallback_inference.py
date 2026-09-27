"""Emergency Full Fallback Inference using Exact Indexed Blocking.

Generates candidates and predicted matches for all 1,732,544 test S1 entities
against all 9,969,589 test target entities using exact normalized/compact
name and address keys restricted to country.
Scores candidates with matcher_model_k40.txt at threshold 0.9000.
Produces:
- output/final_fallback_candidate_pairs.tsv
- output/final_fallback_matching_results.tsv
"""

import gc
import json
import os
import resource
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    OUTPUT_DIR,
    TEST_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from src.features import FEATURE_COLUMNS, build_feature_matrix_v2
from src.inverted_index import get_peak_memory_mb, normalize_country
from src.preprocess import (
    compact_address,
    compact_name,
    extract_numeric_tokens,
    normalize_address,
    normalize_name,
    preprocess_dataframe,
)


def run_full_fallback(
    s1_batch_size: int = 100000,
    target_chunk_size: int = 250000,
    threshold: float = 0.9000,
):
    t_global_start = time.time()

    cand_out = OUTPUT_DIR / "final_fallback_candidate_pairs.tsv"
    match_out = OUTPUT_DIR / "final_fallback_matching_results.tsv"
    model_path = OUTPUT_DIR / "matcher_model_k40.txt"
    db_path = OUTPUT_DIR / "cache" / "test_targets.db"
    checkpoint_dir = OUTPUT_DIR / "fallback_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("EMERGENCY FULL FALLBACK INFERENCE: EXACT INDEXED BLOCKING")
    print("=" * 88)

    # 1. Print total S1 rows, total target rows, estimated runtime
    # Pre-count rows
    total_s1_rows = 1732544
    total_target_rows = 9969589
    estimated_runtime_min = 12.0

    print(f"Total S1 Rows:          {total_s1_rows:,}")
    print(f"Total Target Rows:      {total_target_rows:,} (Source 2: 4,887,274 + Source 3: 5,082,317)")
    print(f"Model Path:             {model_path}")
    print(f"Decision Threshold:     {threshold:.4f}")
    print(f"Estimated Runtime:      ~{estimated_runtime_min:.1f} minutes")
    print(f"Candidate Output:       {cand_out}")
    print(f"Matching Output:        {match_out}")
    print("=" * 88)

    # 2. Check if candidate file already exists and is complete
    cand_checkpoint_path = checkpoint_dir / "all_s1_candidates.tsv"
    need_candidate_retrieval = True
    if cand_checkpoint_path.exists():
        with open(cand_checkpoint_path, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        if lines == total_s1_rows + 1:
            print(f"Found complete candidate checkpoint with {lines:,} lines at {cand_checkpoint_path}!")
            need_candidate_retrieval = False
        else:
            print(f"Incomplete candidate checkpoint ({lines:,} lines). Re-running retrieval...")

    # Load S1 IDs and Data
    print("\n[Phase 1] Loading and normalizing test S1 entities...")
    t0 = time.time()
    s1_df = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
    )
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    assert len(s1_ids) == total_s1_rows, f"Expected {total_s1_rows} rows, got {len(s1_ids)}"
    print(f"Loaded {len(s1_ids):,} test S1 entities in {time.time() - t0:.2f}s | RSS: {get_peak_memory_mb():.1f} MB")

    if need_candidate_retrieval:
        # Pre-normalize S1 entities
        t_norm = time.time()
        s1_names = [normalize_name(x) for x in s1_df["business_name"].fillna("")]
        s1_cnames = [compact_name(x) for x in s1_df["business_name"].fillna("")]
        s1_addrs = [normalize_address(x) for x in s1_df["business_address"].fillna("")]
        s1_caddrs = [compact_address(x) for x in s1_df["business_address"].fillna("")]
        s1_countries = [normalize_country(x) for x in s1_df["country"].fillna("")]
        print(f"Normalized S1 strings in {time.time() - t_norm:.2f}s | RSS: {get_peak_memory_mb():.1f} MB")

        # Build key maps to S1 indices
        print("Indexing S1 exact keys...")
        name_map: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        cname_map: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        addr_map: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        caddr_map: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for idx in range(total_s1_rows):
            c = s1_countries[idx]
            if not c:
                continue
            nm = s1_names[idx]
            if nm:
                name_map[(c, nm)].append(idx)
            cnm = s1_cnames[idx]
            if cnm:
                cname_map[(c, cnm)].append(idx)
            ad = s1_addrs[idx]
            if ad:
                addr_map[(c, ad)].append(idx)
            cad = s1_caddrs[idx]
            if cad:
                caddr_map[(c, cad)].append(idx)

        print(f"Key index ready:")
        print(f"  (country, name_norm):    {len(name_map):,} unique keys")
        print(f"  (country, name_compact): {len(cname_map):,} unique keys")
        print(f"  (country, addr_norm):    {len(addr_map):,} unique keys")
        print(f"  (country, addr_compact): {len(caddr_map):,} unique keys")

        # Allocate candidate lists (indexed by S1 row index)
        s1_candidates: List[List[str]] = [[] for _ in range(total_s1_rows)]

        # Stream all targets in Source 2 and Source 3
        print("\nStreaming test target records (Source 2 & Source 3)...")
        t_stream_start = time.time()
        processed_targets = 0
        total_cand_hits = 0

        for src_name, path in [("Source 2", TEST_SOURCE2), ("Source 3", TEST_SOURCE3)]:
            print(f"Scanning {src_name} ({path.name})...")
            reader = pd.read_csv(
                path,
                sep="\t",
                chunksize=target_chunk_size,
                usecols=["entity_id", "business_name", "business_address", "country"],
                dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
            )
            for chunk in reader:
                chunk_n = len(chunk)
                processed_targets += chunk_n

                eids = chunk["entity_id"].astype(str).tolist()
                bnames = chunk["business_name"].fillna("").tolist()
                baddrs = chunk["business_address"].fillna("").tolist()
                countries = chunk["country"].tolist()

                for eid, bname, baddr, country in zip(eids, bnames, baddrs, countries):
                    c = normalize_country(country)
                    if not c:
                        continue
                    nn = normalize_name(bname)
                    nc = compact_name(bname)
                    an = normalize_address(baddr)
                    ac = compact_address(baddr)

                    matched_s1_indices = set()

                    if nn and (c, nn) in name_map:
                        matched_s1_indices.update(name_map[(c, nn)])
                    if nc and (c, nc) in cname_map:
                        matched_s1_indices.update(cname_map[(c, nc)])
                    if an and (c, an) in addr_map:
                        matched_s1_indices.update(addr_map[(c, an)])
                    if ac and (c, ac) in caddr_map:
                        matched_s1_indices.update(caddr_map[(c, ac)])

                    if matched_s1_indices:
                        for s1_idx in matched_s1_indices:
                            s1_candidates[s1_idx].append(eid)
                        total_cand_hits += len(matched_s1_indices)

                # Periodic log during target scan
                el = time.time() - t_stream_start
                tp = processed_targets / el if el > 0 else 0
                rem_targets = total_target_rows - processed_targets
                eta_s = rem_targets / tp if tp > 0 else 0
                if processed_targets % (target_chunk_size * 2) < target_chunk_size:
                    print(
                        f"  [Targets] {processed_targets:>9,} / {total_target_rows:,} ({processed_targets/total_target_rows*100:5.1f}%) | "
                        f"Cand hits: {total_cand_hits:>9,} | Rate: {tp:8.0f} rows/s | ETA: {eta_s:4.0f}s | RSS: {get_peak_memory_mb():.1f} MB"
                    )

        t_stream_elapsed = time.time() - t_stream_start
        print(f"Target streaming complete in {t_stream_elapsed:.2f}s ({processed_targets/t_stream_elapsed:.0f} rows/s)")

        # Deduplicate candidate lists preserving first-seen order and write checkpoint
        print("\nDeduplicating candidates and saving candidate checkpoint...")
        t_save = time.time()
        zero_cands = 0
        total_unique_cands = 0

        with open(cand_checkpoint_path, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            for idx in range(total_s1_rows):
                sid = s1_ids[idx]
                raw_cands = s1_candidates[idx]
                if not raw_cands:
                    zero_cands += 1
                    f_cand.write(f"{sid}\t\n")
                else:
                    # Deduplicate preserving first-seen order
                    seen = set()
                    deduped = []
                    for cid in raw_cands:
                        if cid not in seen:
                            seen.add(cid)
                            deduped.append(cid)
                    total_unique_cands += len(deduped)
                    f_cand.write(f"{sid}\t{','.join(deduped)}\n")

        print(f"Saved candidate checkpoint in {time.time() - t_save:.2f}s.")
        print(f"Total Unique Candidates: {total_unique_cands:,} (avg {total_unique_cands/total_s1_rows:.2f}/S1)")
        print(f"S1 with Zero Candidates: {zero_cands:,} / {total_s1_rows:,} ({zero_cands/total_s1_rows*100:.2f}%)")

        # Free memory from index maps
        del name_map, cname_map, addr_map, caddr_map, s1_candidates
        gc.collect()

    # Copy candidate checkpoint to final output candidate pairs file
    shutil.copyfile(cand_checkpoint_path, cand_out)
    print(f"Candidate file ready at {cand_out}")

    # =========================================================================
    # Phase 2: Feature Generation & Scoring in Checkpointed Batches
    # =========================================================================
    print("\n" + "=" * 88)
    print("[Phase 2] Batched Feature Engineering & LightGBM Scoring")
    print("=" * 88)

    booster = lgb.Booster(model_file=str(model_path))

    # Read candidate pairs from candidate file
    print("Reading candidate file for batch scoring...")
    s1_cand_map: Dict[str, List[str]] = {}
    with open(cand_checkpoint_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            cands = parts[1].split(",") if (len(parts) > 1 and parts[1]) else []
            s1_cand_map[sid] = cands

    # Connect to SQLite for target attributes
    assert db_path.exists(), f"Target database not found at {db_path}"
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=50000")

    num_batches = (total_s1_rows + s1_batch_size - 1) // s1_batch_size
    print(f"Processing {total_s1_rows:,} S1 entities across {num_batches} batches of {s1_batch_size:,}...")

    t_score_start = time.time()
    s1_processed_total = 0
    total_predicted_matches = 0
    zero_match_s1_count = 0

    shard_match_files: List[Path] = []

    for b_idx in range(num_batches):
        start_idx = b_idx * s1_batch_size
        end_idx = min(start_idx + s1_batch_size, total_s1_rows)
        batch_s1_df = s1_df.iloc[start_idx:end_idx].copy()
        batch_s1_ids = batch_s1_df["entity_id"].tolist()
        batch_n = len(batch_s1_ids)

        shard_path = checkpoint_dir / f"shard_match_batch_{b_idx:03d}.tsv"
        shard_match_files.append(shard_path)

        # Check if shard already complete
        if shard_path.exists():
            with open(shard_path, "r", encoding="utf-8") as sf:
                s_lines = sum(1 for _ in sf)
            if s_lines == batch_n:
                s1_processed_total += batch_n
                # Count matches from shard
                with open(shard_path, "r", encoding="utf-8") as sf:
                    for line in sf:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) > 1 and parts[1]:
                            m_count = len(parts[1].split(","))
                            total_predicted_matches += m_count
                        else:
                            zero_match_s1_count += 1
                continue

        t_b_start = time.time()

        # Collect candidate pairs and needed target IDs for this batch
        batch_pairs_s1 = []
        batch_pairs_cand = []
        batch_needed_targets = set()

        for sid in batch_s1_ids:
            cands = s1_cand_map.get(sid, [])
            for cid in cands:
                batch_pairs_s1.append(sid)
                batch_pairs_cand.append(cid)
                batch_needed_targets.add(cid)

        # Retrieve target attributes from SQLite
        target_dict: Dict[str, dict] = {}
        if batch_needed_targets:
            ids_list = list(batch_needed_targets)
            SQL_CHUNK = 900
            for i in range(0, len(ids_list), SQL_CHUNK):
                chunk_ids = ids_list[i : i + SQL_CHUNK]
                ph = ",".join("?" * len(chunk_ids))
                rows = con.execute(
                    f"SELECT entity_id, business_name, business_address, country FROM targets WHERE entity_id IN ({ph})",
                    chunk_ids,
                ).fetchall()
                for eid, bname, baddr, country in rows:
                    nn = normalize_name(bname)
                    na = normalize_address(baddr)
                    target_dict[str(eid)] = {
                        "entity_id": str(eid),
                        "name_normalized": nn,
                        "name_compact": nn.replace(" ", ""),
                        "address_normalized": na,
                        "address_compact": na.replace(" ", ""),
                        "address_numeric_tokens": extract_numeric_tokens(baddr),
                        "country_norm_lc": normalize_country(country),
                    }

        # Pre-normalize S1 batch
        batch_norm_df = preprocess_dataframe(batch_s1_df, include_token_lists=False)
        batch_norm_df["country_norm_lc"] = [normalize_country(c) for c in batch_norm_df["country"]]
        _raw_addr_col = "business_address" if "business_address" in batch_norm_df.columns else "address"
        _raw_addrs = batch_norm_df[_raw_addr_col].fillna("").tolist()
        batch_norm_df["address_numeric_tokens"] = [extract_numeric_tokens(x) for x in _raw_addrs]
        s1_lookup = batch_norm_df.set_index("entity_id").to_dict(orient="index")

        matched_map: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in batch_s1_ids}

        n_pairs = len(batch_pairs_s1)
        if n_pairs > 0:
            cs_arr = np.zeros(n_pairs, dtype=np.float32)
            ws_arr = np.zeros(n_pairs, dtype=np.float32)
            feat_df = build_feature_matrix_v2(
                s1_ids_list=batch_pairs_s1,
                cand_ids_list=batch_pairs_cand,
                char_sims=cs_arr,
                word_sims=ws_arr,
                s1_dict=s1_lookup,
                target_dict=target_dict,
            )
            probs = booster.predict(feat_df[FEATURE_COLUMNS].values)
            for (sid, cid), prob in zip(zip(batch_pairs_s1, batch_pairs_cand), probs):
                if prob >= threshold:
                    matched_map[sid].append((cid, float(prob)))

        # Write shard file (deterministic confidence sort, subset invariant guaranteed)
        with open(shard_path, "w", encoding="utf-8") as sf:
            for sid in batch_s1_ids:
                m_list = matched_map[sid]
                if not m_list:
                    zero_match_s1_count += 1
                    sf.write(f"{sid}\t\n")
                else:
                    # Sort descending by probability
                    sorted_cids = [cid for cid, _ in sorted(m_list, key=lambda x: -x[1])]
                    # Deduplicate preserving order
                    seen_cids = set()
                    final_cids = []
                    for cid in sorted_cids:
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            final_cids.append(cid)
                    total_predicted_matches += len(final_cids)
                    sf.write(f"{sid}\t{','.join(final_cids)}\n")

        s1_processed_total += batch_n

        # Periodically report progress
        elapsed = time.time() - t_score_start
        s1_rate = s1_processed_total / elapsed if elapsed > 0 else 0
        rem_s1 = total_s1_rows - s1_processed_total
        eta_s = rem_s1 / s1_rate if s1_rate > 0 else 0

        print(
            f"  [Scoring Batch {b_idx+1:>2}/{num_batches}] S1 Completed: {s1_processed_total:>9,} | "
            f"Remaining: {rem_s1:>9,} | Rate: {s1_rate:6.1f} S1/s | "
            f"Matches: {total_predicted_matches:>8,} | ETA: {eta_s:4.0f}s | RSS: {get_peak_memory_mb():.1f} MB"
        )

    con.close()

    # =========================================================================
    # Phase 3: Final Merge & Strict Output Verification
    # =========================================================================
    print("\n" + "=" * 88)
    print("[Phase 3] Deterministic Merge & Final Output Verification")
    print("=" * 88)

    t_merge_start = time.time()
    with open(match_out, "w", encoding="utf-8") as mf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        merged_s1_count = 0
        for shard_p in shard_match_files:
            with open(shard_p, "r", encoding="utf-8") as sf:
                for line in sf:
                    mf.write(line)
                    merged_s1_count += 1

    assert merged_s1_count == total_s1_rows, f"Merged {merged_s1_count} S1 rows, expected {total_s1_rows}"
    print(f"Merged {merged_s1_count:,} matching rows into {match_out} in {time.time() - t_merge_start:.2f}s.")

    # Count total candidate rows
    with open(cand_out, "r", encoding="utf-8") as cf:
        cand_line_count = sum(1 for _ in cf) - 1
    assert cand_line_count == total_s1_rows, f"Candidate file has {cand_line_count} rows, expected {total_s1_rows}"

    # Count total unique candidates and zero-candidate S1
    total_candidates = 0
    zero_cand_s1_count = 0
    with open(cand_out, "r", encoding="utf-8") as cf:
        cf.readline()  # skip header
        for line in cf:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1 and parts[1]:
                total_candidates += len(parts[1].split(","))
            else:
                zero_cand_s1_count += 1

    total_runtime = time.time() - t_global_start

    print("\n" + "=" * 88)
    print("EXECUTION SUMMARY:")
    print("=" * 88)
    print(f"Total Runtime:                    {total_runtime:.2f}s ({total_runtime/60:.2f} minutes)")
    print(f"Total S1 Entities:                {total_s1_rows:,}")
    print(f"Total Candidate Pairs:            {total_candidates:,} (avg {total_candidates/total_s1_rows:.2f}/S1)")
    print(f"Total Predicted Matches:          {total_predicted_matches:,} (avg {total_predicted_matches/total_s1_rows:.4f}/S1)")
    print(f"S1 with Zero Candidates:          {zero_cand_s1_count:,} / {total_s1_rows:,} ({zero_cand_s1_count/total_s1_rows*100:.2f}%)")
    print(f"S1 with Zero Predicted Matches:   {zero_match_s1_count:,} / {total_s1_rows:,} ({zero_match_s1_count/total_s1_rows*100:.2f}%)")
    print(f"Candidate File:                   {cand_out} ({cand_out.stat().st_size / 1e6:.1f} MB)")
    print(f"Matching File:                    {match_out} ({match_out.stat().st_size / 1e6:.1f} MB)")
    print(f"Peak Memory (RSS):                {get_peak_memory_mb():.1f} MB")
    print("=" * 88)


if __name__ == "__main__":
    run_full_fallback()
