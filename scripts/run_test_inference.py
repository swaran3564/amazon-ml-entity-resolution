"""Production Test Inference Pipeline for Amazon ML Challenge 2026 Entity Resolution.

Locked Architecture:
1. Candidate Retrieval:
   - Character TF-IDF Top-40 (char (3, 5) n-grams)
   - Word TF-IDF Top-40 (word (1, 2) n-grams)
   - Candidate UNION (~57 candidates/S1)
   - Dynamic country partitioning with normalized_country
   - Explicit fallback mechanism for missing/empty/null country
2. Pairwise Feature Engineering:
   - 22 numerical features via src.features.build_feature_matrix
   - emb_sim placeholder maintained
3. Model Scoring:
   - LightGBM Booster: output/matcher_model_k40.txt
4. Threshold & Decision Rule:
   - Global Decision Threshold = 0.9000
   - Multi-match allowed (zero, one, or many matches per S1)
5. Sharded Output & Deterministic Merging:
   - Separate temporary output files per worker/shard (never concurrent writes to same TSV)
   - Deterministic final merge enforcing:
     * Exactly one candidate row per test S1
     * Exactly one matching row per test S1
     * Deduplicated candidate IDs within each S1 list
     * Deduplicated matched IDs within each S1 list
     * Strict subset constraint: matching IDs must be subset of candidate IDs
     * Preserves empty rows for S1s with zero candidates/matches
"""

import argparse
import gc
import heapq
import multiprocessing as mp
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocking import _extract_normalized_names
from src.config import (
    CHAR_TFIDF_TOP_K,
    OUTPUT_CANDIDATE,
    OUTPUT_DIR,
    OUTPUT_MATCHING,
    TEST_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    WORD_TFIDF_TOP_K,
)
from src.features import FEATURE_COLUMNS, build_feature_matrix, build_feature_matrix_v2
from src.inverted_index import (
    build_or_load_country_inverted_index,
    retrieve_candidates_from_index,
)
from src.preprocess import (
    compact_address,
    compact_name,
    extract_numeric_tokens,
    normalize_address,
    normalize_name,
    preprocess_dataframe,
)


def normalize_country(country: Any) -> str:
    """Safely normalize a country string via strip and casefold.

    Returns '' for missing/empty/null/NaN representations.
    Do NOT hard-code country mappings.
    """
    if country is None or pd.isna(country):
        return ""
    s = str(country).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s.casefold()


def get_peak_memory_mb() -> float:
    """Get peak resident set size of current process in megabytes."""
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw_rss / (1024 * 1024))
    return float(raw_rss / 1024)


def load_model_and_vectorizers(
    model_path: Path,
    cache_dir: Path,
) -> Tuple[lgb.Booster, Any, Any]:
    """Load trained LightGBM booster model and cached TF-IDF vectorizers."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained LightGBM model not found at {model_path}")

    booster = lgb.Booster(model_file=str(model_path))

    char_vec_path = cache_dir / "char_vectorizer.joblib"
    word_vec_path = cache_dir / "word_vectorizer.joblib"

    if not char_vec_path.exists() or not word_vec_path.exists():
        raise FileNotFoundError(
            f"Pre-fitted global vectorizers not found in {cache_dir}. "
            f"Expected char_vectorizer.joblib and word_vectorizer.joblib."
        )

    char_vec = joblib.load(char_vec_path)
    word_vec = joblib.load(word_vec_path)

    return booster, char_vec, word_vec


def stream_target_chunks_with_country(
    s2_path: Path,
    s3_path: Path,
    chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
) -> Generator[Tuple[pd.DataFrame, str], None, None]:
    """Stream target candidate DataFrames in bounded chunks from Source 2 and Source 3 TSV files with country."""
    total_yielded = 0
    for source_label, file_path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not file_path.exists():
            continue
        reader = pd.read_csv(
            file_path,
            sep="\t",
            usecols=["entity_id", "business_name", "country"],
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


def retrieve_union_candidates_for_batch(
    s1_batch_df: pd.DataFrame,
    char_vec: Any,
    word_vec: Any,
    s2_path: Path,
    s3_path: Path,
    char_top_k: int = 40,
    word_top_k: int = 40,
    target_chunk_size: int = 100000,
    max_target_rows: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Retrieve Character Top-K and Word Top-K candidates for a batch of S1 queries with Country Partitioning & Fallback.

    Returns:
        candidate_pairs_map: {sid: [cid_1, cid_2, ...]} (ranked by combined score)
        char_scores: {sid: {cid: char_sim}}
        word_scores: {sid: {cid: word_sim}}
    """
    s1_ids = s1_batch_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_batch_df)
    s1_countries = [normalize_country(c) for c in s1_batch_df["country"].tolist()]

    queries_by_country: Dict[str, List[Tuple[int, str]]] = {}
    fallback_queries: List[Tuple[int, str]] = []  # S1 with missing/empty country

    for idx, (sid, norm_c) in enumerate(zip(s1_ids, s1_countries)):
        if norm_c == "":
            fallback_queries.append((idx, sid))
        else:
            queries_by_country.setdefault(norm_c, []).append((idx, sid))

    # Vectorize S1 queries
    X_s1_char = char_vec.transform(s1_names).tocsr()
    X_s1_word = word_vec.transform(s1_names).tocsr()

    char_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    word_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}

    chunk_stream = stream_target_chunks_with_country(
        s2_path=s2_path,
        s3_path=s3_path,
        chunk_size=target_chunk_size,
        max_target_rows=max_target_rows,
    )

    for chunk_df, _ in chunk_stream:
        target_ids = np.array(chunk_df["entity_id"].astype(str).tolist())
        target_names = _extract_normalized_names(chunk_df)
        target_countries = np.array([normalize_country(c) for c in chunk_df["country"].tolist()])

        # Vectorize chunk once
        X_chunk_char = char_vec.transform(target_names).tocsr()
        X_chunk_word = word_vec.transform(target_names).tocsr()

        unique_countries = np.unique(target_countries)

        # 1. Evaluate matching country partitions
        for c_val in unique_countries:
            if c_val == "":
                continue  # handled in target fallback below

            if c_val not in queries_by_country and not fallback_queries:
                continue

            c_mask = (target_countries == c_val)
            c_indices = np.where(c_mask)[0]
            if len(c_indices) == 0:
                continue

            c_target_ids = target_ids[c_mask]
            X_c_char_T = X_chunk_char[c_mask].T.tocsr()
            X_c_word_T = X_chunk_word[c_mask].T.tocsr()

            # Queries for this country + any fallback queries (queries with missing country)
            eval_queries = queries_by_country.get(c_val, []) + fallback_queries

            # Character TF-IDF
            for orig_idx, sid in eval_queries:
                q_vec = X_s1_char[orig_idx : orig_idx + 1]
                if q_vec.nnz > 0:
                    sims = q_vec.dot(X_c_char_T).tocsr()
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

            # Word TF-IDF
            for orig_idx, sid in eval_queries:
                q_vec = X_s1_word[orig_idx : orig_idx + 1]
                if q_vec.nnz > 0:
                    sims = q_vec.dot(X_c_word_T).tocsr()
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

        # 2. Target Fallback: any target with missing/empty country is evaluated against ALL S1 queries
        if "" in unique_countries:
            fb_mask = (target_countries == "")
            fb_target_ids = target_ids[fb_mask]
            X_fb_char_T = X_chunk_char[fb_mask].T.tocsr()
            X_fb_word_T = X_chunk_word[fb_mask].T.tocsr()

            for orig_idx, sid in enumerate(s1_ids):
                # Char
                q_vec = X_s1_char[orig_idx : orig_idx + 1]
                if q_vec.nnz > 0:
                    sims = q_vec.dot(X_fb_char_T).tocsr()
                    if sims.nnz > 0:
                        data, cols = sims.data, sims.indices
                        s1_map = char_cands[sid]
                        for col, score in zip(cols, data):
                            cid = fb_target_ids[col]
                            s = float(score)
                            if cid not in s1_map or s > s1_map[cid]:
                                s1_map[cid] = s
                # Word
                q_vec = X_s1_word[orig_idx : orig_idx + 1]
                if q_vec.nnz > 0:
                    sims = q_vec.dot(X_fb_word_T).tocsr()
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

    # Form final UNION candidates and sort deterministically
    candidate_pairs_map: Dict[str, List[str]] = {}
    final_char_scores: Dict[str, Dict[str, float]] = {}
    final_word_scores: Dict[str, Dict[str, float]] = {}

    for sid in s1_ids:
        c_top = heapq.nsmallest(char_top_k, char_cands[sid].items(), key=lambda x: (-x[1], x[0]))
        w_top = heapq.nsmallest(word_top_k, word_cands[sid].items(), key=lambda x: (-x[1], x[0]))
        c_map = dict(c_top)
        w_map = dict(w_top)

        final_char_scores[sid] = c_map
        final_word_scores[sid] = w_map

        # Union and rank candidates deterministically by combined similarity, then candidate entity_id
        combined: Dict[str, float] = {}
        for cid, s in c_map.items():
            combined[cid] = combined.get(cid, 0.0) + s
        for cid, s in w_map.items():
            combined[cid] = combined.get(cid, 0.0) + s

        ranked_cands = [k for k, _ in sorted(combined.items(), key=lambda x: (-x[1], x[0]))]
        candidate_pairs_map[sid] = ranked_cands

    return candidate_pairs_map, final_char_scores, final_word_scores


def _collect_target_records_fast(
    candidate_ids: Set[str],
    s2_path: Path,
    s3_path: Path,
    chunk_size: int = 200000,
) -> Dict[str, dict]:
    """Vectorized target-attribute collection with pre-normalisation.

    Replaces iterrows() + lazy normalisation with:
    - Vectorized pandas filter (isin mask)
    - Single-pass normalisation at collection time
    - Pre-builds all keys required by build_feature_matrix_v2:
        name_normalized, name_compact, address_normalized, address_compact,
        address_numeric_tokens, country_norm_lc
    """
    target_dict: Dict[str, dict] = {}
    needed = set(candidate_ids)

    for src_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not path.exists() or not needed:
            continue
        reader = pd.read_csv(
            path,
            sep="\t",
            chunksize=chunk_size,
            usecols=["entity_id", "business_name", "business_address", "country"],
        )
        for chunk in reader:
            # Vectorized filter — no iterrows
            chunk_ids = chunk["entity_id"].astype(str)
            mask = chunk_ids.isin(needed)
            if not mask.any():
                continue
            matches = chunk[mask]
            eids = chunk_ids[mask].tolist()
            bnames = matches["business_name"].tolist()
            baddrs = matches["business_address"].tolist()
            countries = matches["country"].tolist()

            for eid, bname, baddr, country in zip(eids, bnames, baddrs, countries):
                name_norm = normalize_name(bname)
                addr_norm = normalize_address(baddr)
                c_norm = normalize_country(country)
                target_dict[eid] = {
                    "entity_id": eid,
                    "name_normalized": name_norm,
                    "name_compact": name_norm.replace(" ", ""),
                    "address_normalized": addr_norm,
                    "address_compact": addr_norm.replace(" ", ""),
                    "address_numeric_tokens": extract_numeric_tokens(baddr),
                    "country_norm_lc": c_norm,
                }
                needed.discard(eid)
            if not needed:
                break

    return target_dict


def _collect_target_records_sqlite(
    candidate_ids: Set[str],
    db_path: Path,
) -> Dict[str, dict]:
    """Fast O(1) target-attribute lookup via the pre-built SQLite index.

    9-10x faster than streaming S2/S3 TSVs: replaces a ~10s full-file scan
    with a ~1s indexed SELECT IN query for ~30K IDs.
    Applies the same normalisation as _collect_target_records_fast so the
    result is a drop-in replacement for build_feature_matrix_v2.
    """
    target_dict: Dict[str, dict] = {}
    if not db_path.exists() or not candidate_ids:
        return target_dict

    ids_list = list(candidate_ids)
    CHUNK = 900  # SQLite variable limit is 999

    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA cache_size=50000")

        for i in range(0, len(ids_list), CHUNK):
            chunk = ids_list[i : i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = con.execute(
                f"SELECT entity_id, business_name, business_address, country"
                f" FROM targets WHERE entity_id IN ({ph})",
                chunk,
            ).fetchall()
            for eid, bname, baddr, country in rows:
                name_norm = normalize_name(bname)
                addr_norm = normalize_address(baddr)
                c_norm = normalize_country(country)
                target_dict[str(eid)] = {
                    "entity_id": str(eid),
                    "name_normalized": name_norm,
                    "name_compact": name_norm.replace(" ", ""),
                    "address_normalized": addr_norm,
                    "address_compact": addr_norm.replace(" ", ""),
                    "address_numeric_tokens": extract_numeric_tokens(baddr),
                    "country_norm_lc": c_norm,
                }
        con.close()
    except Exception as exc:
        print(f"[SQLite lookup failed: {exc}] — result may be partial", flush=True)

    return target_dict



def collect_target_records_for_ids(
    candidate_ids: Set[str],
    s2_path: Path,
    s3_path: Path,
    chunk_size: int = 200000,
) -> Dict[str, dict]:
    """Scan Source 2 and Source 3 files to retrieve business attributes for candidate target IDs."""
    target_dict: Dict[str, dict] = {}
    needed = set(candidate_ids)

    for src_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not path.exists() or not needed:
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

    return target_dict


def merge_worker_outputs(
    shard_cand_paths: List[Path],
    shard_match_paths: List[Path],
    required_s1_ids: Sequence[str],
    final_cand_path: Path,
    final_match_path: Path,
) -> Dict[str, Any]:
    """Deterministically reconcile and merge shard outputs into final TSV files.

    Guarantees:
    - exactly one candidate row per required S1
    - exactly one matching row per required S1
    - deterministic ordering matching required_s1_ids
    - deduplication of candidate IDs within each S1 list (preserving first-seen order)
    - deduplication of matched IDs within each S1 list (preserving confidence order)
    - strict subset invariant: matching IDs must be a subset of candidate IDs
    - preservation of empty rows for S1s with zero candidates/matches
    - zero self-matches (S1- prefix) or non-S2/S3 IDs
    """
    cand_map: Dict[str, List[str]] = {str(sid): [] for sid in required_s1_ids}
    match_map: Dict[str, List[str]] = {str(sid): [] for sid in required_s1_ids}

    # 1. Read shard candidate files
    for p in shard_cand_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                s1, tab, rest = line.partition("\t")
                if not tab:
                    continue
                s1 = s1.strip()
                if s1 in cand_map:
                    raw_ids = [cid.strip() for cid in rest.rstrip("\n").split(",") if cid.strip()]
                    cand_map[s1].extend(raw_ids)

    # 2. Read shard matching files
    for p in shard_match_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                s1, tab, rest = line.partition("\t")
                if not tab:
                    continue
                s1 = s1.strip()
                if s1 in match_map:
                    raw_ids = [mid.strip() for mid in rest.rstrip("\n").split(",") if mid.strip()]
                    match_map[s1].extend(raw_ids)

    # 3. Deterministically reconcile, deduplicate, and enforce invariants
    final_cand_path.parent.mkdir(parents=True, exist_ok=True)
    final_match_path.parent.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    total_matches = 0
    zero_cand_s1 = 0
    zero_match_s1 = 0
    subset_violations = 0

    with open(final_cand_path, "w", encoding="utf-8") as f_cand, open(final_match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for sid in required_s1_ids:
            sid = str(sid)

            # Deduplicate candidate IDs (preserving order, only S2-/S3-)
            seen_cands: Set[str] = set()
            dedup_cands: List[str] = []
            for cid in cand_map.get(sid, []):
                if cid.startswith(("S2-", "S3-")) and cid not in seen_cands:
                    seen_cands.add(cid)
                    dedup_cands.append(cid)

            # Deduplicate match IDs and enforce strict subset of candidates
            seen_matches: Set[str] = set()
            dedup_matches: List[str] = []
            for mid in match_map.get(sid, []):
                if mid.startswith(("S2-", "S3-")) and mid in seen_cands and mid not in seen_matches:
                    seen_matches.add(mid)
                    dedup_matches.append(mid)

            if len(dedup_cands) == 0:
                zero_cand_s1 += 1
            if len(dedup_matches) == 0:
                zero_match_s1 += 1

            total_candidates += len(dedup_cands)
            total_matches += len(dedup_matches)

            # Write outputs
            f_cand.write(f"{sid}\t{','.join(dedup_cands)}\n")
            f_match.write(f"{sid}\t{','.join(dedup_matches)}\n")

    return {
        "total_s1_rows": len(required_s1_ids),
        "total_candidate_ids": total_candidates,
        "total_predicted_matches": total_matches,
        "zero_candidate_s1_count": zero_cand_s1,
        "zero_match_s1_count": zero_match_s1,
        "subset_violations": subset_violations,
    }


def _worker_process_shard(
    worker_id: int,
    s1_shard_df: pd.DataFrame,
    s2_path: str,
    s3_path: str,
    model_path: str,
    cache_dir: str,
    out_cand_path: str,
    out_match_path: str,
    threshold: float,
    char_top_k: int,
    word_top_k: int,
    target_chunk_size: int,
    s1_chunk_size: int,
    max_target_rows: Optional[int],
    use_index: Optional[bool] = None,
    stop_ngrams_path: Optional[str] = None,
    use_stop_ngrams: bool = True,
) -> None:
    """Worker task that processes a shard of S1 entities and writes to private shard TSVs."""
    s2_p = Path(s2_path)
    s3_p = Path(s3_path)
    model_p = Path(model_path)
    cache_p = Path(cache_dir)
    cand_p = Path(out_cand_path)
    match_p = Path(out_match_path)

    booster, char_vec, word_vec = load_model_and_vectorizers(model_p, cache_p)

    cand_p.parent.mkdir(parents=True, exist_ok=True)
    match_p.parent.mkdir(parents=True, exist_ok=True)

    with open(cand_p, "w", encoding="utf-8") as f_c, open(match_p, "w", encoding="utf-8") as f_m:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    # Determine whether to use cached country inverted index
    index_test_dir = cache_p / "index_test"
    if use_index is None:
        use_index = (index_test_dir / "_SUCCESS").exists()

    country_indexes = None
    if use_index:
        try:
            print(f"[Worker {worker_id}] Loading country inverted indexes from {index_test_dir}...", flush=True)
            country_indexes = build_or_load_country_inverted_index(
                split_name="test",
                s2_path=s2_p,
                s3_path=s3_p,
                char_vec=char_vec,
                word_vec=word_vec,
                cache_dir=cache_p,
            )
            print(f"[Worker {worker_id}] Inverted indexes ready.", flush=True)
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to load inverted index ({e}). Falling back to streaming.", flush=True)
            use_index = False
            country_indexes = None

    stop_ngrams_arg = Path(stop_ngrams_path) if stop_ngrams_path else None

    n_total = len(s1_shard_df)
    for start_idx in range(0, n_total, s1_chunk_size):
        sub_df = s1_shard_df.iloc[start_idx : start_idx + s1_chunk_size].copy()

        if use_index and country_indexes is not None:
            cands_map, char_scores, word_scores = retrieve_candidates_from_index(
                s1_df=sub_df,
                country_indexes=country_indexes,
                char_vec=char_vec,
                word_vec=word_vec,
                char_top_k=char_top_k,
                word_top_k=word_top_k,
                stop_ngrams=stop_ngrams_arg,
                use_stop_ngrams=use_stop_ngrams,
            )
        else:
            cands_map, char_scores, word_scores = retrieve_union_candidates_for_batch(
                s1_batch_df=sub_df,
                char_vec=char_vec,
                word_vec=word_vec,
                s2_path=s2_p,
                s3_path=s3_p,
                char_top_k=char_top_k,
                word_top_k=word_top_k,
                target_chunk_size=target_chunk_size,
                max_target_rows=max_target_rows,
            )

        batch_pairs_s1: List[str] = []
        batch_pairs_cand: List[str] = []
        batch_char_sims: List[float] = []
        batch_word_sims: List[float] = []
        batch_cand_ids: Set[str] = set()
        for sid, c_list in cands_map.items():
            c_scores = char_scores[sid]
            w_scores = word_scores[sid]
            for cid in c_list:
                batch_pairs_s1.append(sid)
                batch_pairs_cand.append(cid)
                batch_char_sims.append(c_scores.get(cid, 0.0))
                batch_word_sims.append(w_scores.get(cid, 0.0))
                batch_cand_ids.add(cid)

        # Prefer SQLite indexed lookup (~1s) over full TSV scan (~10s).
        _db_path = cache_p / "test_targets.db"
        if _db_path.exists():
            target_dict = _collect_target_records_sqlite(batch_cand_ids, _db_path)
            # If any IDs are missing from DB, fall back to TSV scan for remainder
            missing = batch_cand_ids - set(target_dict.keys())
            if missing:
                fallback = _collect_target_records_fast(missing, s2_p, s3_p)
                target_dict.update(fallback)
        else:
            target_dict = _collect_target_records_fast(batch_cand_ids, s2_p, s3_p)

        # Pre-normalise S1 once per sub-batch.
        # preprocess_dataframe adds: name_normalized, name_compact,
        #   address_normalized, address_compact, country_normalized (upper).
        sub_df_normed = preprocess_dataframe(sub_df, include_token_lists=False)
        sub_df_normed["country_norm_lc"] = [
            normalize_country(c) for c in sub_df_normed["country"]
        ]
        # Numeric tokens from raw address column (before normalization)
        _raw_addr_col = "business_address" if "business_address" in sub_df.columns else "address"
        _raw_addrs = sub_df[_raw_addr_col].tolist() if _raw_addr_col in sub_df.columns else [""] * len(sub_df)
        sub_df_normed["address_numeric_tokens"] = [extract_numeric_tokens(x) for x in _raw_addrs]
        s1_lookup = sub_df_normed.set_index("entity_id").to_dict(orient="index")

        matched_map: Dict[str, List[str]] = {str(sid): [] for sid in sub_df["entity_id"]}

        if batch_pairs_s1:
            cs_arr = np.array(batch_char_sims, dtype=np.float32)
            ws_arr = np.array(batch_word_sims, dtype=np.float32)
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

            for sid in matched_map:
                matched_map[sid] = [cid for cid, _ in sorted(matched_map[sid], key=lambda x: -x[1])]

        with open(cand_p, "a", encoding="utf-8") as f_c:
            for sid in sub_df["entity_id"].astype(str):
                f_c.write(f"{sid}\t{','.join(cands_map.get(sid, []))}\n")

        with open(match_p, "a", encoding="utf-8") as f_m:
            for sid in sub_df["entity_id"].astype(str):
                f_m.write(f"{sid}\t{','.join(matched_map.get(sid, []))}\n")


def run_test_inference(
    s1_path: Path = TEST_SOURCE1,
    s2_path: Path = TEST_SOURCE2,
    s3_path: Path = TEST_SOURCE3,
    model_path: Path = OUTPUT_DIR / "matcher_model_k40.txt",
    cache_dir: Path = OUTPUT_DIR / "cache",
    candidate_output_path: Path = OUTPUT_CANDIDATE,
    matching_output_path: Path = OUTPUT_MATCHING,
    threshold: float = 0.9000,
    char_top_k: int = 40,
    word_top_k: int = 40,
    target_chunk_size: int = 100000,
    s1_chunk_size: int = 2000,
    max_s1_rows: Optional[int] = None,
    max_target_rows: Optional[int] = None,
    num_workers: int = 1,
    use_index: Optional[bool] = None,
    stop_ngrams_path: Optional[Path] = None,
    use_stop_ngrams: bool = True,
) -> Dict[str, Any]:
    """Execute test inference pipeline with worker isolation and deterministic merging."""
    t_start = time.time()
    print("=" * 88)
    print("AMAZON ML CHALLENGE 2026: TEST INFERENCE PIPELINE (K=40)")
    print("=" * 88)
    print(f"Source 1 Test File:     {s1_path}")
    print(f"Source 2 Test File:     {s2_path}")
    print(f"Source 3 Test File:     {s3_path}")
    print(f"Matcher Model:          {model_path}")
    print(f"Decision Threshold:     {threshold:.4f}")
    print(f"Candidate Cutoff:       Char Top-{char_top_k} UNION Word Top-{word_top_k} (Country-Partitioned)")
    print(f"Candidate Output:       {candidate_output_path}")
    print(f"Matching Output:        {matching_output_path}")
    print(f"Parallel Workers:       {num_workers}")
    print(f"Stop N-grams Enabled:   {use_stop_ngrams} ({stop_ngrams_path or 'auto-loaded'})")
    print("=" * 88)

    # Read S1 queries
    s1_full_df = pd.read_csv(s1_path, sep="\t")
    if max_s1_rows is not None:
        s1_full_df = s1_full_df.iloc[:max_s1_rows].copy()

    s1_ids = s1_full_df["entity_id"].astype(str).tolist()
    n_s1 = len(s1_ids)
    print(f"Total S1 entities to process: {n_s1:,}")

    # Set up temporary shard directory
    temp_dir = Path(tempfile.mkdtemp(prefix="inference_shards_"))

    try:
        shard_cand_paths: List[Path] = []
        shard_match_paths: List[Path] = []

        if num_workers <= 1:
            # Single-worker execution
            shard_cand = temp_dir / "worker_0_candidates.tsv"
            shard_match = temp_dir / "worker_0_matches.tsv"
            shard_cand_paths.append(shard_cand)
            shard_match_paths.append(shard_match)

            _worker_process_shard(
                worker_id=0,
                s1_shard_df=s1_full_df,
                s2_path=str(s2_path),
                s3_path=str(s3_path),
                model_path=str(model_path),
                cache_dir=str(cache_dir),
                out_cand_path=str(shard_cand),
                out_match_path=str(shard_match),
                threshold=threshold,
                char_top_k=char_top_k,
                word_top_k=word_top_k,
                target_chunk_size=target_chunk_size,
                s1_chunk_size=s1_chunk_size,
                max_target_rows=max_target_rows,
                use_index=use_index,
                stop_ngrams_path=str(stop_ngrams_path) if stop_ngrams_path else None,
                use_stop_ngrams=use_stop_ngrams,
            )
            peak_rss = get_peak_memory_mb()
        else:
            # Multi-worker execution with process isolation
            shards = [s1_full_df.iloc[i::num_workers].copy() for i in range(num_workers)]
            processes: List[mp.Process] = []

            for w_idx, shard_df in enumerate(shards):
                sc_p = temp_dir / f"worker_{w_idx}_candidates.tsv"
                sm_p = temp_dir / f"worker_{w_idx}_matches.tsv"
                shard_cand_paths.append(sc_p)
                shard_match_paths.append(sm_p)

                p = mp.Process(
                    target=_worker_process_shard,
                    args=(
                        w_idx,
                        shard_df,
                        str(s2_path),
                        str(s3_path),
                        str(model_path),
                        str(cache_dir),
                        str(sc_p),
                        str(sm_p),
                        threshold,
                        char_top_k,
                        word_top_k,
                        target_chunk_size,
                        s1_chunk_size,
                        max_target_rows,
                        use_index,
                        str(stop_ngrams_path) if stop_ngrams_path else None,
                        use_stop_ngrams,
                    ),
                )
                processes.append(p)
                p.start()

            # Monitor memory across parent and child worker processes
            peak_total_rss = 0.0
            parent_proc = psutil.Process()

            while any(p.is_alive() for p in processes):
                try:
                    total_rss = parent_proc.memory_info().rss
                    for child in parent_proc.children(recursive=True):
                        total_rss += child.memory_info().rss
                    rss_mb = total_rss / (1024 * 1024)
                    if rss_mb > peak_total_rss:
                        peak_total_rss = rss_mb
                except Exception:
                    pass
                time.sleep(0.5)

            for p in processes:
                p.join()

            peak_rss = peak_total_rss if peak_total_rss > 0.0 else get_peak_memory_mb()

        # Deterministic reconciliation and final merge
        print(f"\nMerging shard outputs into final submission files...")
        merge_stats = merge_worker_outputs(
            shard_cand_paths=shard_cand_paths,
            shard_match_paths=shard_match_paths,
            required_s1_ids=s1_ids,
            final_cand_path=candidate_output_path,
            final_match_path=matching_output_path,
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_runtime = time.time() - t_start
    cand_size_bytes = candidate_output_path.stat().st_size if candidate_output_path.exists() else 0
    match_size_bytes = matching_output_path.stat().st_size if matching_output_path.exists() else 0

    results = {
        "candidate_pairs_tsv_size_bytes": cand_size_bytes,
        "matching_results_tsv_size_bytes": match_size_bytes,
        "number_of_test_s1_entities": merge_stats["total_s1_rows"],
        "total_candidate_ids_generated": merge_stats["total_candidate_ids"],
        "total_predicted_matches": merge_stats["total_predicted_matches"],
        "s1_entities_with_zero_predicted_matches": merge_stats["zero_match_s1_count"],
        "s1_entities_with_zero_candidates": merge_stats["zero_candidate_s1_count"],
        "avg_candidates_per_s1": float(merge_stats["total_candidate_ids"] / max(1, merge_stats["total_s1_rows"])),
        "avg_predicted_matches_per_s1": float(merge_stats["total_predicted_matches"] / max(1, merge_stats["total_s1_rows"])),
        "subset_violations": merge_stats["subset_violations"],
        "runtime_seconds": float(round(total_runtime, 2)),
        "peak_memory_mb": float(round(peak_rss, 2)),
        "num_workers": num_workers,
    }

    print("\n" + "=" * 88)
    print(f"INFERENCE SUMMARY ({num_workers} Worker{'s' if num_workers > 1 else ''}):")
    print("=" * 88)
    print(f"Number of Test S1 Entities Processed:    {results['number_of_test_s1_entities']:,}")
    print(f"Total Candidate IDs Generated:          {results['total_candidate_ids_generated']:,}")
    print(f"Average Candidates per S1:              {results['avg_candidates_per_s1']:.2f}")
    print(f"Total Predicted Matches:                {results['total_predicted_matches']:,}")
    print(f"Average Predicted Matches per S1:       {results['avg_predicted_matches_per_s1']:.4f}")
    print(f"S1 Entities with 0 Predicted Matches:   {results['s1_entities_with_zero_predicted_matches']:,}")
    print(f"S1 Entities with 0 Candidates:          {results['s1_entities_with_zero_candidates']:,}")
    print(f"candidate_pairs.tsv Size:               {cand_size_bytes / 1e6:.2f} MB")
    print(f"matching_results.tsv Size:              {match_size_bytes / 1e6:.2f} MB")
    print(f"Runtime:                                {total_runtime:.2f}s ({total_runtime / 60:.2f} min)")
    print(f"Peak Total Memory (RSS):                {peak_rss:.2f} MB")
    print("=" * 88)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Test Inference Pipeline.")
    parser.add_argument("--s1-path", type=Path, default=TEST_SOURCE1, help="Path to test_source1.tsv")
    parser.add_argument("--s2-path", type=Path, default=TEST_SOURCE2, help="Path to test_source2.tsv")
    parser.add_argument("--s3-path", type=Path, default=TEST_SOURCE3, help="Path to test_source3.tsv")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "matcher_model_k40.txt", help="Trained booster model path")
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT_DIR / "cache", help="Cache directory with vectorizers")
    parser.add_argument("--candidate-output", type=Path, default=OUTPUT_CANDIDATE, help="Output candidate_pairs.tsv path")
    parser.add_argument("--matching-output", type=Path, default=OUTPUT_MATCHING, help="Output matching_results.tsv path")
    parser.add_argument("--threshold", type=float, default=0.9000, help="Decision threshold (default: 0.9000)")
    parser.add_argument("--char-top-k", type=int, default=40, help="Character Top-K cutoff (default: 40)")
    parser.add_argument("--word-top-k", type=int, default=40, help="Word Top-K cutoff (default: 40)")
    parser.add_argument("--target-chunk-size", type=int, default=100000, help="Target chunk size (default: 100000)")
    parser.add_argument("--s1-chunk-size", type=int, default=2000, help="S1 batch size (default: 2000)")
    parser.add_argument("--max-s1-rows", type=int, default=None, help="Optional max S1 rows to process")
    parser.add_argument("--max-target-rows", type=int, default=None, help="Optional max target rows to scan")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes (default: 1)")
    parser.add_argument(
        "--use-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cached country inverted index if available (default: True if index exists)",
    )

    parser.add_argument("--stop-ngrams-path", type=Path, default=None, help="Path to stop n-grams JSON config")
    parser.add_argument(
        "--use-stop-ngrams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable high-frequency stop n-gram pruning (default: True)",
    )

    args = parser.parse_args()

    run_test_inference(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        candidate_output_path=args.candidate_output,
        matching_output_path=args.matching_output,
        threshold=args.threshold,
        char_top_k=args.char_top_k,
        word_top_k=args.word_top_k,
        target_chunk_size=args.target_chunk_size,
        s1_chunk_size=args.s1_chunk_size,
        max_s1_rows=args.max_s1_rows,
        max_target_rows=args.max_target_rows,
        num_workers=args.num_workers,
        use_index=args.use_index,
        stop_ngrams_path=args.stop_ngrams_path,
        use_stop_ngrams=args.use_stop_ngrams,
    )


if __name__ == "__main__":
    main()

