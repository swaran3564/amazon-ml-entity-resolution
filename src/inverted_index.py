"""Inverted index and sparse blocking module for Amazon ML Challenge 2026.

Provides:
- Country-partitioned inverted indexes for Character and Word TF-IDF target representations.
- Mathematically exact sparse candidate retrieval (only targets with non-zero similarity are considered).
- Zero lexical pruning threshold (no arbitrary similarity cutoff; exact Top-K per branch).
- Strict deterministic entity-ID tie-breaking.
- Automatic caching and fast loading of index structures.
- Fallback routing for queries or targets with missing/empty country.
"""

import gc
import heapq
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocking import _extract_normalized_names

DEFAULT_STOP_NGRAMS_PATH = REPO_ROOT / "output" / "cache" / "stop_ngrams.json"


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


def build_or_load_country_inverted_index(
    split_name: str,
    s2_path: Path,
    s3_path: Path,
    char_vec: Any,
    word_vec: Any,
    cache_dir: Path,
    chunk_size: int = 500000,
    force_rebuild: bool = False,
) -> Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]]:
    """Build or load per-country inverted index (transposed CSR matrices) for target corpus.

    For each country partition C:
    - char_index: CSR matrix of shape (V_char, N_C)
    - word_index: CSR matrix of shape (V_word, N_C)
    - target_ids: 1D numpy array of entity IDs of length N_C

    Returns:
        {country_name: (char_index, word_index, target_ids)}
    """
    index_dir = cache_dir / f"index_{split_name}"
    index_dir.mkdir(parents=True, exist_ok=True)
    done_marker = index_dir / "_SUCCESS"

    if done_marker.exists() and not force_rebuild:
        print(f"Loading cached {split_name} target inverted indexes from {index_dir}...")
        t0 = time.time()
        indexes: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]] = {}
        for country_dir in sorted(index_dir.iterdir()):
            if country_dir.is_dir():
                c_val = country_dir.name
                c_char_path = country_dir / "char_index.npz"
                c_word_path = country_dir / "word_index.npz"
                c_ids_path = country_dir / "target_ids.npy"

                # Check if memory-mappable .npy arrays are available for zero-copy fast loading
                c_char_indptr = country_dir / "char_indptr.npy"
                c_char_indices = country_dir / "char_indices.npy"
                c_char_data = country_dir / "char_data.npy"
                c_word_indptr = country_dir / "word_indptr.npy"
                c_word_indices = country_dir / "word_indices.npy"
                c_word_data = country_dir / "word_data.npy"

                if (
                    c_char_indptr.exists()
                    and c_char_indices.exists()
                    and c_char_data.exists()
                    and c_word_indptr.exists()
                    and c_word_indices.exists()
                    and c_word_data.exists()
                    and c_ids_path.exists()
                ):
                    t_c0 = time.time()
                    c_ids = np.load(c_ids_path, allow_pickle=True, mmap_mode="r")
                    n_targets = len(c_ids)

                    char_indptr = np.load(c_char_indptr, mmap_mode="r")
                    char_indices = np.load(c_char_indices, mmap_mode="r")
                    char_data = np.load(c_char_data, mmap_mode="r")
                    c_char = sp.csr_matrix(
                        (char_data, char_indices, char_indptr),
                        shape=(len(char_indptr) - 1, n_targets),
                    )

                    word_indptr = np.load(c_word_indptr, mmap_mode="r")
                    word_indices = np.load(c_word_indices, mmap_mode="r")
                    word_data = np.load(c_word_data, mmap_mode="r")
                    c_word = sp.csr_matrix(
                        (word_data, word_indices, word_indptr),
                        shape=(len(word_indptr) - 1, n_targets),
                    )

                    indexes[c_val] = (c_char, c_word, c_ids)
                    print(f"  Loaded country '{c_val}' (mmap): {len(c_ids):,} targets in {time.time() - t_c0:.3f}s")
                elif c_char_path.exists() and c_word_path.exists() and c_ids_path.exists():
                    t_c0 = time.time()
                    c_char = sp.load_npz(c_char_path)
                    c_word = sp.load_npz(c_word_path)
                    c_ids = np.load(c_ids_path, allow_pickle=True)
                    indexes[c_val] = (c_char, c_word, c_ids)
                    print(f"  Loaded country '{c_val}': {len(c_ids):,} targets in {time.time() - t_c0:.2f}s")
        if indexes:
            print(f"All {split_name} inverted indexes loaded in {time.time() - t0:.2f}s.")
            return indexes

    print(f"Building {split_name} target inverted indexes from {s2_path.name} and {s3_path.name}...")
    t_build_start = time.time()

    char_chunks: Dict[str, List[sp.csr_matrix]] = {}
    word_chunks: Dict[str, List[sp.csr_matrix]] = {}
    id_chunks: Dict[str, List[np.ndarray]] = {}

    total_rows = 0
    for label, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        if not path.exists():
            continue
        reader = pd.read_csv(
            path,
            sep="\t",
            usecols=["entity_id", "business_name", "country"],
            chunksize=chunk_size,
        )
        for chunk_df in reader:
            n = len(chunk_df)
            total_rows += n

            ids = np.array(chunk_df["entity_id"].astype(str).tolist())
            names = _extract_normalized_names(chunk_df)
            countries = np.array([normalize_country(c) for c in chunk_df["country"].tolist()])

            # Vectorize chunk
            X_char = char_vec.transform(names).tocsr()
            X_word = word_vec.transform(names).tocsr()

            for c_val in np.unique(countries):
                mask = (countries == c_val)
                char_chunks.setdefault(c_val, []).append(X_char[mask])
                word_chunks.setdefault(c_val, []).append(X_word[mask])
                id_chunks.setdefault(c_val, []).append(ids[mask])

            rss = psutil.Process().memory_info().rss / 1e6
            el = time.time() - t_build_start
            print(f"  Streamed {total_rows:>10,} targets in {el:.1f}s | RSS: {rss:.1f} MB")

    # Finalize per-country inverted indexes
    indexes = {}
    for c_val in sorted(char_chunks.keys()):
        print(f"Finalizing inverted index for country partition '{c_val}'...")
        c_ids = np.concatenate(id_chunks[c_val])
        c_char = sp.vstack(char_chunks[c_val], format="csr").T.tocsr()
        c_word = sp.vstack(word_chunks[c_val], format="csr").T.tocsr()

        c_dir = index_dir / c_val
        c_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(c_dir / "char_index.npz", c_char)
        sp.save_npz(c_dir / "word_index.npz", c_word)
        np.save(c_dir / "target_ids.npy", c_ids)

        indexes[c_val] = (c_char, c_word, c_ids)
        print(f"  Country '{c_val}': {len(c_ids):,} targets saved to {c_dir}")

        del char_chunks[c_val]
        del word_chunks[c_val]
        del id_chunks[c_val]
        gc.collect()

    done_marker.touch()
    print(f"Inverted index construction completed in {time.time() - t_build_start:.2f}s!")
    return indexes


def load_stop_ngrams_config(
    config_path_or_dict: Optional[Union[str, Path, Dict[str, Dict[str, Sequence[str]]]]] = None
) -> Dict[str, Dict[str, List[str]]]:
    """Load stop n-grams configuration from a file or dict.

    Returns:
        {country: {"char": [stop_char_ngrams], "word": [stop_word_ngrams]}}
    """
    if config_path_or_dict is None:
        if DEFAULT_STOP_NGRAMS_PATH.exists():
            with open(DEFAULT_STOP_NGRAMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    if isinstance(config_path_or_dict, (str, Path)):
        p = Path(config_path_or_dict)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    if isinstance(config_path_or_dict, dict):
        return {
            str(c).lower(): {
                "char": list(v.get("char", [])),
                "word": list(v.get("word", [])),
            }
            for c, v in config_path_or_dict.items()
        }
    return {}


def extract_corpus_stop_ngrams(
    country_indexes: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]],
    char_vec: Any,
    word_vec: Any,
    df_threshold: Union[int, float, Dict[str, Union[int, float]]] = 500000,
) -> Dict[str, Dict[str, List[str]]]:
    """Extract high-frequency corporate stop n-grams from target index document frequencies.

    Guarantees:
    - Stop n-grams are determined purely from measured corpus statistics.
    - Zero hardcoded English/business knowledge.
    - Fully configurable by count threshold, ratio, or per-country dictionary.
    """
    char_names = np.array(char_vec.get_feature_names_out())
    word_names = np.array(word_vec.get_feature_names_out())

    result: Dict[str, Dict[str, List[str]]] = {}

    for c_val, (c_char_idx, c_word_idx, c_tids) in country_indexes.items():
        n_targets = len(c_tids)
        if n_targets == 0:
            continue

        if isinstance(df_threshold, dict):
            c_thresh = df_threshold.get(c_val, None)
        else:
            c_thresh = df_threshold

        if c_thresh is None:
            continue

        if isinstance(c_thresh, float) and 0.0 < c_thresh < 1.0:
            thresh_count = int(c_thresh * n_targets)
        else:
            thresh_count = int(c_thresh)

        char_dfs = np.diff(c_char_idx.indptr)
        word_dfs = np.diff(c_word_idx.indptr)

        char_stops = sorted(char_names[char_dfs > thresh_count].tolist())
        word_stops = sorted(word_names[word_dfs > thresh_count].tolist())

        if char_stops or word_stops:
            result[c_val] = {
                "char": char_stops,
                "word": word_stops,
            }

    return result


try:
    import numba
    from numba import prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(fastmath=True)
    def _score_query_numba_buf(
        q_feats, q_weights, is_stop, indptr, indices, data, scores, touched_targets, touched_scores, top_k
    ):
        n_touched = 0
        # Check if query has at least one non-stop feature
        has_non_stop = False
        for i in range(len(q_feats)):
            if not is_stop[q_feats[i]]:
                has_non_stop = True
                break

        for i in range(len(q_feats)):
            feat = q_feats[i]
            if has_non_stop and is_stop[feat]:
                continue
            w = q_weights[i]
            start = indptr[feat]
            end = indptr[feat + 1]
            for j in range(start, end):
                tid = indices[j]
                if scores[tid] == 0.0:
                    touched_targets[n_touched] = tid
                    n_touched += 1
                scores[tid] += w * data[j]

        if n_touched <= top_k:
            top_indices = touched_targets[:n_touched].copy()
            top_scores = np.empty(n_touched, dtype=np.float32)
            for i in range(n_touched):
                top_scores[i] = scores[top_indices[i]]
        else:
            for i in range(n_touched):
                touched_scores[i] = scores[touched_targets[i]]
            sub_scores = touched_scores[:n_touched]
            part_idx = np.argpartition(sub_scores, -top_k)[-top_k:]
            top_indices = np.empty(top_k, dtype=np.int32)
            top_scores = np.empty(top_k, dtype=np.float32)
            for i in range(top_k):
                idx = part_idx[i]
                top_indices[i] = touched_targets[idx]
                top_scores[i] = sub_scores[idx]

        # Clean up scores for next query
        for i in range(n_touched):
            scores[touched_targets[i]] = 0.0

        return top_indices, top_scores

    @numba.njit(parallel=True, fastmath=True)
    def _score_batch_numba_parallel(
        q_orig_indices,
        q_indptr, q_indices, q_data,
        is_stop,
        idx_indptr, idx_indices, idx_data,
        scores_buf, touched_buf, touched_sc_buf,
        n_threads, top_k,
        out_top_indices, out_top_scores, out_actual_k,
    ):
        n_q = len(q_orig_indices)
        for t in prange(n_threads):
            start_q = (t * n_q) // n_threads
            end_q = ((t + 1) * n_q) // n_threads

            scores = scores_buf[t]
            touched = touched_buf[t]
            touched_sc = touched_sc_buf[t]

            for q_idx in range(start_q, end_q):
                orig_row = q_orig_indices[q_idx]
                q_start = q_indptr[orig_row]
                q_end = q_indptr[orig_row + 1]

                # Check if query has at least one non-stop feature
                has_non_stop = False
                for i in range(q_start, q_end):
                    feat = q_indices[i]
                    if not is_stop[feat]:
                        has_non_stop = True
                        break

                n_touched = 0
                for i in range(q_start, q_end):
                    feat = q_indices[i]
                    if has_non_stop and is_stop[feat]:
                        continue
                    w = q_data[i]
                    p_start = idx_indptr[feat]
                    p_end = idx_indptr[feat + 1]
                    for j in range(p_start, p_end):
                        tid = idx_indices[j]
                        if scores[tid] == 0.0:
                            touched[n_touched] = tid
                            n_touched += 1
                        scores[tid] += w * idx_data[j]

                if n_touched <= top_k:
                    out_actual_k[q_idx] = n_touched
                    for i in range(n_touched):
                        tid = touched[i]
                        out_top_indices[q_idx, i] = tid
                        out_top_scores[q_idx, i] = scores[tid]
                else:
                    out_actual_k[q_idx] = top_k
                    for i in range(n_touched):
                        touched_sc[i] = scores[touched[i]]
                    sub_scores = touched_sc[:n_touched]
                    part_idx = np.argpartition(sub_scores, -top_k)[-top_k:]
                    for i in range(top_k):
                        p_i = part_idx[i]
                        out_top_indices[q_idx, i] = touched[p_i]
                        out_top_scores[q_idx, i] = sub_scores[p_i]

                # Clean up scores for next query
                for i in range(n_touched):
                    scores[touched[i]] = 0.0


def retrieve_candidates_from_index(
    s1_df: pd.DataFrame,
    country_indexes: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]],
    char_vec: Any,
    word_vec: Any,
    char_top_k: int = 40,
    word_top_k: int = 40,
    stop_ngrams: Optional[Union[Dict[str, Dict[str, Sequence[str]]], Path, str]] = None,
    stop_ngram_df_threshold: Optional[Union[int, float, Dict[str, Union[int, float]]]] = None,
    use_stop_ngrams: bool = True,
    n_threads: int = 4,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Retrieve Character Top-K and Word Top-K candidates for S1 queries using the inverted index.

    Guarantees:
    - Pure sparse matrix operations (only non-zero overlaps are scored).
    - Top-K exact cosine similarity for character branch and word branch.
    - Deterministic tie-breaking on (-score, cid).
    - Ranked UNION candidate list per S1.
    - Bidirectional fallback for queries with missing/empty country.
    - High-frequency corporate stop n-gram pruning on posting traversal.

    Returns:
        candidate_pairs_map: {sid: [cid1, cid2, ...]} (ranked by combined score)
        char_scores: {sid: {cid: char_sim}}
        word_scores: {sid: {cid: word_sim}}
    """
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_df)
    s1_countries = [normalize_country(c) for c in s1_df["country"].tolist()]

    # Resolve stop n-grams configuration
    resolved_stop_ngrams: Dict[str, Dict[str, List[str]]] = {}
    if use_stop_ngrams:
        if stop_ngrams is not None:
            resolved_stop_ngrams = load_stop_ngrams_config(stop_ngrams)
        elif stop_ngram_df_threshold is not None:
            resolved_stop_ngrams = extract_corpus_stop_ngrams(
                country_indexes, char_vec, word_vec, df_threshold=stop_ngram_df_threshold
            )
        else:
            resolved_stop_ngrams = load_stop_ngrams_config(None)

    # Pre-build boolean stop masks per country
    char_vocab = getattr(char_vec, "vocabulary_", None)
    word_vocab = getattr(word_vec, "vocabulary_", None)

    is_stop_char_by_country: Dict[str, np.ndarray] = {}
    is_stop_word_by_country: Dict[str, np.ndarray] = {}

    v_char_len = len(char_vocab) if char_vocab is not None else (
        next(iter(country_indexes.values()))[0].shape[0] if country_indexes else 0
    )
    v_word_len = len(word_vocab) if word_vocab is not None else (
        next(iter(country_indexes.values()))[1].shape[0] if country_indexes else 0
    )

    for c_val, (c_char_idx, c_word_idx, _) in country_indexes.items():
        mask_c = np.zeros(c_char_idx.shape[0], dtype=np.bool_)
        mask_w = np.zeros(c_word_idx.shape[0], dtype=np.bool_)

        if resolved_stop_ngrams and c_val in resolved_stop_ngrams:
            c_stops = resolved_stop_ngrams[c_val]
            if char_vocab is not None:
                for s in c_stops.get("char", []):
                    if s in char_vocab:
                        mask_c[char_vocab[s]] = True
            if word_vocab is not None:
                for s in c_stops.get("word", []):
                    if s in word_vocab:
                        mask_w[word_vocab[s]] = True

            n_cs = int(mask_c.sum())
            n_ws = int(mask_w.sum())
            if n_cs > 0 or n_ws > 0:
                print(f"  [Country '{c_val}'] Pruning {n_cs} char stop n-grams and {n_ws} word stop n-grams (total {n_cs + n_ws})", flush=True)

        is_stop_char_by_country[c_val] = mask_c
        is_stop_word_by_country[c_val] = mask_w

    empty_stop_char = np.zeros(v_char_len, dtype=np.bool_)
    empty_stop_word = np.zeros(v_word_len, dtype=np.bool_)

    # Group queries by country
    queries_by_country: Dict[str, List[Tuple[int, str]]] = {}
    fallback_queries: List[Tuple[int, str]] = []

    for idx, (sid, c) in enumerate(zip(s1_ids, s1_countries)):
        if c == "":
            fallback_queries.append((idx, sid))
        else:
            queries_by_country.setdefault(c, []).append((idx, sid))

    # Vectorize S1 queries
    X_s1_char = char_vec.transform(s1_names).tocsr()
    X_s1_word = word_vec.transform(s1_names).tocsr()

    char_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
    word_cands: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}

    # Allocate reusable scratch buffers for parallel Numba JIT
    scores_buf = None
    touched_buf = None
    touched_sc_buf = None
    effective_threads = max(1, n_threads)
    if HAS_NUMBA and country_indexes:
        max_targets = max(tids.shape[0] for _, _, tids in country_indexes.values())
        scores_buf = np.zeros((effective_threads, max_targets), dtype=np.float32)
        touched_buf = np.empty((effective_threads, max_targets), dtype=np.int32)
        touched_sc_buf = np.empty((effective_threads, max_targets), dtype=np.float32)
        # Warmup JIT compile
        dummy_q_idx = np.array([0], dtype=np.int32)
        dummy_q_ptr = np.array([0, 1], dtype=np.int32)
        dummy_q_feat = np.array([0], dtype=np.int32)
        dummy_q_dat = np.array([1.0], dtype=np.float32)
        dummy_stop = np.array([False], dtype=np.bool_)
        dummy_ptr = np.array([0, 1], dtype=np.int32)
        dummy_idx = np.array([0], dtype=np.int32)
        dummy_dat = np.array([1.0], dtype=np.float32)
        dummy_out_idx = np.empty((1, 5), dtype=np.int32)
        dummy_out_sc = np.empty((1, 5), dtype=np.float32)
        dummy_out_k = np.empty(1, dtype=np.int32)
        _score_batch_numba_parallel(
            dummy_q_idx, dummy_q_ptr, dummy_q_feat, dummy_q_dat,
            dummy_stop, dummy_ptr, dummy_idx, dummy_dat,
            scores_buf, touched_buf, touched_sc_buf, 1, 5,
            dummy_out_idx, dummy_out_sc, dummy_out_k
        )

    def _query_partition(
        eval_queries: List[Tuple[int, str]],
        c_char_index: sp.csr_matrix,
        c_word_index: sp.csr_matrix,
        c_target_ids: np.ndarray,
        is_stop_char: np.ndarray,
        is_stop_word: np.ndarray,
    ):
        n_queries = len(eval_queries)
        if n_queries == 0:
            return

        top_k_char = char_top_k * 2
        top_k_word = word_top_k * 2

        if HAS_NUMBA and scores_buf is not None:
            # Batch queries in chunks to balance cache locality and intermediate reporting
            batch_chunk_size = 2000
            for start_i in range(0, n_queries, batch_chunk_size):
                end_i = min(start_i + batch_chunk_size, n_queries)
                chunk_queries = eval_queries[start_i:end_i]
                chunk_len = len(chunk_queries)
                q_orig_indices = np.array([idx for idx, sid in chunk_queries], dtype=np.int32)
                chunk_sids = [sid for idx, sid in chunk_queries]
                threads_to_use = min(effective_threads, chunk_len)

                # 1. Character TF-IDF
                out_top_indices_c = np.empty((chunk_len, top_k_char), dtype=np.int32)
                out_top_scores_c = np.empty((chunk_len, top_k_char), dtype=np.float32)
                out_actual_k_c = np.empty(chunk_len, dtype=np.int32)

                _score_batch_numba_parallel(
                    q_orig_indices,
                    X_s1_char.indptr, X_s1_char.indices, X_s1_char.data,
                    is_stop_char,
                    c_char_index.indptr, c_char_index.indices, c_char_index.data,
                    scores_buf, touched_buf, touched_sc_buf,
                    threads_to_use, top_k_char,
                    out_top_indices_c, out_top_scores_c, out_actual_k_c,
                )

                for q_i in range(chunk_len):
                    sid = chunk_sids[q_i]
                    k_c = out_actual_k_c[q_i]
                    if k_c > 0:
                        top_idx = out_top_indices_c[q_i, :k_c]
                        top_sc = out_top_scores_c[q_i, :k_c]
                        cids = c_target_ids[top_idx]
                        s1_map = char_cands[sid]
                        for cid, score in zip(cids, top_sc):
                            sc = float(score)
                            if cid not in s1_map or sc > s1_map[cid]:
                                s1_map[cid] = sc

                # 2. Word TF-IDF
                out_top_indices_w = np.empty((chunk_len, top_k_word), dtype=np.int32)
                out_top_scores_w = np.empty((chunk_len, top_k_word), dtype=np.float32)
                out_actual_k_w = np.empty(chunk_len, dtype=np.int32)

                _score_batch_numba_parallel(
                    q_orig_indices,
                    X_s1_word.indptr, X_s1_word.indices, X_s1_word.data,
                    is_stop_word,
                    c_word_index.indptr, c_word_index.indices, c_word_index.data,
                    scores_buf, touched_buf, touched_sc_buf,
                    threads_to_use, top_k_word,
                    out_top_indices_w, out_top_scores_w, out_actual_k_w,
                )

                for q_i in range(chunk_len):
                    sid = chunk_sids[q_i]
                    k_w = out_actual_k_w[q_i]
                    if k_w > 0:
                        top_idx_w = out_top_indices_w[q_i, :k_w]
                        top_sc_w = out_top_scores_w[q_i, :k_w]
                        cids_w = c_target_ids[top_idx_w]
                        s1_map_w = word_cands[sid]
                        for cid, score in zip(cids_w, top_sc_w):
                            sc = float(score)
                            if cid not in s1_map_w or sc > s1_map_w[cid]:
                                s1_map_w[cid] = sc

                if n_queries >= 500 and (end_i % 2000 == 0 or end_i == n_queries):
                    print(f"    Queried {end_i:,}/{n_queries:,} ({end_i/n_queries*100:.1f}%) | RSS: {get_peak_memory_mb():.1f} MB", flush=True)

        else:
            # Fallback for systems without Numba
            char_indptr = c_char_index.indptr
            char_indices = c_char_index.indices
            char_data = c_char_index.data

            word_indptr = c_word_index.indptr
            word_indices = c_word_index.indices
            word_data = c_word_index.data

            for count, (orig_idx, sid) in enumerate(eval_queries, 1):
                if n_queries >= 500 and (count % 2000 == 0 or count == n_queries):
                    print(f"    Queried {count:,}/{n_queries:,} ({count/n_queries*100:.1f}%) | RSS: {get_peak_memory_mb():.1f} MB", flush=True)

                qc = X_s1_char[orig_idx]
                if qc.nnz > 0:
                    has_non_stop = bool((~is_stop_char[qc.indices]).any())
                    if has_non_stop and is_stop_char[qc.indices].any():
                        keep_mask = ~is_stop_char[qc.indices]
                        qc_eval = sp.csr_matrix(
                            (qc.data[keep_mask], qc.indices[keep_mask], np.array([0, keep_mask.sum()])),
                            shape=qc.shape,
                        )
                    else:
                        qc_eval = qc

                    sims_char = qc_eval.dot(c_char_index)
                    d = sims_char.data
                    cols = sims_char.indices
                    if len(d) > top_k_char:
                        p = np.argpartition(d, -top_k_char)[-top_k_char:]
                        d = d[p]
                        cols = cols[p]
                    top_idx = cols
                    top_sc = d

                    s1_map = char_cands[sid]
                    cids = c_target_ids[top_idx]
                    for cid, score in zip(cids, top_sc):
                        sc = float(score)
                        if cid not in s1_map or sc > s1_map[cid]:
                            s1_map[cid] = sc

                qw = X_s1_word[orig_idx]
                if qw.nnz > 0:
                    has_non_stop_w = bool((~is_stop_word[qw.indices]).any())
                    if has_non_stop_w and is_stop_word[qw.indices].any():
                        keep_mask_w = ~is_stop_word[qw.indices]
                        qw_eval = sp.csr_matrix(
                            (qw.data[keep_mask_w], qw.indices[keep_mask_w], np.array([0, keep_mask_w.sum()])),
                            shape=qw.shape,
                        )
                    else:
                        qw_eval = qw

                    sims_word = qw_eval.dot(c_word_index)
                    d_w = sims_word.data
                    cols_w = sims_word.indices
                    if len(d_w) > top_k_word:
                        p_w = np.argpartition(d_w, -top_k_word)[-top_k_word:]
                        d_w = d_w[p_w]
                        cols_w = cols_w[p_w]
                    top_idx_w = cols_w
                    top_sc_w = d_w

                    s1_map_w = word_cands[sid]
                    cids_w = c_target_ids[top_idx_w]
                    for cid, score in zip(cids_w, top_sc_w):
                        sc = float(score)
                        if cid not in s1_map_w or sc > s1_map_w[cid]:
                            s1_map_w[cid] = sc

    # 1. Query matching country partitions
    for c_val, q_list in queries_by_country.items():
        if c_val in country_indexes:
            print(f"  Retrieving candidates for country '{c_val}' ({len(q_list):,} S1 queries)...", flush=True)
            c_char_idx, c_word_idx, c_tids = country_indexes[c_val]
            m_c = is_stop_char_by_country.get(c_val, empty_stop_char)
            m_w = is_stop_word_by_country.get(c_val, empty_stop_word)
            _query_partition(q_list, c_char_idx, c_word_idx, c_tids, m_c, m_w)

    # 2. Fallback routing for queries with missing/empty country
    if fallback_queries:
        print(f"  Retrieving candidates for fallback queries ({len(fallback_queries):,} S1)...", flush=True)
        for c_val, (c_char_idx, c_word_idx, c_tids) in country_indexes.items():
            m_c = is_stop_char_by_country.get(c_val, empty_stop_char)
            m_w = is_stop_word_by_country.get(c_val, empty_stop_word)
            _query_partition(fallback_queries, c_char_idx, c_word_idx, c_tids, m_c, m_w)

    # 3. Target fallback partition (if any target had missing country)
    if "" in country_indexes:
        fb_char_idx, fb_word_idx, fb_tids = country_indexes[""]
        all_queries = [(idx, sid) for idx, sid in enumerate(s1_ids)]
        _query_partition(all_queries, fb_char_idx, fb_word_idx, fb_tids, empty_stop_char, empty_stop_word)

    # 4. Form final Top-K and UNION per S1
    candidate_pairs_map: Dict[str, List[str]] = {}
    final_char_scores: Dict[str, Dict[str, float]] = {}
    final_word_scores: Dict[str, Dict[str, float]] = {}

    for sid in s1_ids:
        # Top-K Character candidates sorted by (-score, cid)
        c_top = heapq.nsmallest(char_top_k, char_cands[sid].items(), key=lambda x: (-x[1], x[0]))
        # Top-K Word candidates sorted by (-score, cid)
        w_top = heapq.nsmallest(word_top_k, word_cands[sid].items(), key=lambda x: (-x[1], x[0]))

        final_char_scores[sid] = dict(c_top)
        final_word_scores[sid] = dict(w_top)

        # Combined score for UNION ranking
        combined: Dict[str, float] = {}
        for cid, sc in c_top:
            combined[cid] = combined.get(cid, 0.0) + sc
        for cid, sc in w_top:
            combined[cid] = combined.get(cid, 0.0) + sc

        ranked_cands = [k for k, _ in sorted(combined.items(), key=lambda x: (-x[1], x[0]))]
        candidate_pairs_map[sid] = ranked_cands

    return candidate_pairs_map, final_char_scores, final_word_scores
