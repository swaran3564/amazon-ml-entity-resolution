"""Candidate generation / blocking module.

Locked candidate-generation architecture:
1. Character TF-IDF retrieval (implemented)
2. Word TF-IDF retrieval (to be implemented in future task)
3. Embedding retrieval (from src/embeddings.py, to be implemented in future task)
Candidate UNION: Combines candidates from all three retrieval channels.
"""

import heapq
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from src.config import (
        CHAR_TFIDF_TOP_K,
        WORD_NGRAM_RANGE,
        WORD_TFIDF_TOP_K,
    )
except ImportError:
    CHAR_TFIDF_TOP_K = 20
    WORD_TFIDF_TOP_K = 20
    WORD_NGRAM_RANGE = (1, 2)

from src.preprocess import normalize_name


def _extract_normalized_names(df: pd.DataFrame) -> List[str]:
    """Extract or compute normalized business names from a DataFrame.

    Robustly handles missing values (None, np.nan, pd.NA, float('nan'))
    and null-string representations ('nan', 'none', 'null', '') across both
    'name_normalized' and 'business_name' columns.
    """
    if "name_normalized" in df.columns:
        names: List[str] = []
        for val in df["name_normalized"]:
            if val is None or pd.isna(val):
                names.append("")
            else:
                s = str(val).strip()
                if s.lower() in ("nan", "none", "null", ""):
                    names.append("")
                else:
                    names.append(s)
        return names
    elif "business_name" in df.columns:
        names: List[str] = []
        for val in df["business_name"]:
            if val is None or pd.isna(val):
                names.append("")
            else:
                s_val = str(val).strip()
                if s_val.lower() in ("nan", "none", "null", ""):
                    names.append("")
                else:
                    norm = normalize_name(val)
                    names.append(norm if norm is not None else "")
        return names
    else:
        raise ValueError("DataFrame must contain 'name_normalized' or 'business_name' column.")


def retrieve_char_tfidf_candidates(
    s1_df: pd.DataFrame,
    target_df: Union[pd.DataFrame, List[pd.DataFrame]],
    top_k: int = CHAR_TFIDF_TOP_K,
    ngram_range: Tuple[int, int] = (3, 5),
    chunk_size: int = 500,
    min_similarity: float = 0.0,
    return_scores: bool = False,
    min_df: int = 1,
) -> Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """Retrieve candidate target IDs for each S1 entity using Character n-gram TF-IDF similarity.

    Memory-safe for multi-million-row target datasets:
    1. Uses float32 TF-IDF matrix representation to cut memory by 50%.
    2. Keeps sparse TF-IDF matrices; never calls toarray() or todense().
    3. Retains chunked S1 query vectorization for efficient batch string transforms.
    4. Computes similarity row-by-row against precomputed transposed target matrix (X_target_T).
    5. Discards intermediate row similarities immediately after extracting Top-K candidates.
    6. Explicitly ensures CSR format for row-wise Top-K extraction.
    7. Deterministically preserves the highest score if duplicate target entity IDs exist.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 query DataFrame containing 'entity_id' and 'business_name' or 'name_normalized'.
    target_df : Union[pd.DataFrame, List[pd.DataFrame]]
        Target candidate DataFrame (or list of DataFrames for Source 2 and Source 3)
        containing 'entity_id' and 'business_name' or 'name_normalized'.
    top_k : int
        Maximum number of candidates to retrieve per S1 entity.
    ngram_range : Tuple[int, int]
        Character n-gram range for TF-IDF (defaults to empirically observed range (3, 5)).
    chunk_size : int
        Number of S1 query records processed per batch to bound memory usage.
    min_similarity : float
        Minimum cosine similarity threshold to consider a candidate.
    return_scores : bool
        If True, returns Dict[s1_id, Dict[cand_id, score]].
        If False, returns Dict[s1_id, Set[cand_id]].
    min_df : int
        Minimum document frequency for character n-grams. Defaults to 1.
        In entity resolution blocking, min_df=1 is strictly appropriate:
        rare character patterns, typos, and unique business names often appear
        only once in the target corpus. Setting min_df >= 2 discards singleton
        n-grams from the vocabulary, resulting in severe recall loss for rare entities.

    Returns
    -------
    Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]]
        Mapping from source1_entity_id to candidate entity IDs (preserving S2- and S3- prefixes).
    """
    if "entity_id" not in s1_df.columns:
        raise ValueError("s1_df must contain 'entity_id' column.")

    # Combine target dataframes if provided as a list
    if isinstance(target_df, list):
        target_df = pd.concat(target_df, ignore_index=True)

    if "entity_id" not in target_df.columns:
        raise ValueError("target_df must contain 'entity_id' column.")

    target_ids = np.array(target_df["entity_id"].astype(str).tolist())
    target_names = _extract_normalized_names(target_df)

    n_targets = len(target_names)
    if n_targets == 0:
        return {str(sid): ({} if return_scores else set()) for sid in s1_df["entity_id"]}

    has_duplicate_targets = len(target_ids) != len(set(target_ids))

    # Fit TF-IDF on target candidate corpus with float32 dtype to halve memory usage
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        dtype=np.float32,
    )
    X_target = vectorizer.fit_transform(target_names)
    if not isinstance(X_target, sp.csr_matrix):
        X_target = X_target.tocsr()

    # Precompute transpose once.
    # X_target is N_target x D (CSR).
    # X_target.T is D x N_target (CSC). Calling .tocsr() produces a D x N_target CSR matrix.
    #
    # MEMORY-SAFETY RATIONALE FOR ROW-WISE TOP-K EXTRACTION:
    # -----------------------------------------------------
    # In multi-million-row target datasets, materializing a full chunk-by-target similarity
    # matrix (e.g., X_chunk.dot(X_target.T)) creates an intermediate sparse matrix of shape
    # (chunk_size, N_target). Even in sparse format, because character 3-5 n-grams have high
    # frequency and broad vocabulary overlap, 500 query rows simultaneously matching targets
    # can produce tens of millions of non-zero entries (nnz), consuming hundreds of megabytes
    # or gigabytes of transient RAM per chunk.
    #
    # By contrast, keeping chunked S1 query vectorization (for fast batch string transform)
    # but computing similarities row-by-row:
    #     row_sims = q_vec.dot(X_target_T).tocsr()  # where q_vec is 1 x D CSR
    # produces a 1 x N_target CSR vector containing similarities ONLY for that single query.
    # Top-K candidate extraction is performed immediately in O(nnz_query) time, and the
    # single-row vector is discarded before processing the next query.
    #
    # This bounds transient similarity memory to O(N_target + max_nnz_single_query) rather
    # than O(chunk_size * nnz_single_query), completely eliminating chunk-wide memory spikes
    # while preserving sparse representations and never invoking dense toarray()/todense().
    X_target_T = X_target.T.tocsr()

    # Process S1 queries in chunks to bound memory
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_df)
    n_s1 = len(s1_ids)

    candidates: Dict[str, Union[Set[str], Dict[str, float]]] = {}

    for start_idx in range(0, n_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_s1)
        chunk_texts = s1_names[start_idx:end_idx]
        chunk_ids = s1_ids[start_idx:end_idx]

        # Chunked sparse query vectorization
        X_chunk = vectorizer.transform(chunk_texts)
        if not isinstance(X_chunk, sp.csr_matrix):
            X_chunk = X_chunk.tocsr()

        for i, s1_id in enumerate(chunk_ids):
            q_vec = X_chunk[i : i + 1]  # 1 x D CSR vector

            if q_vec.nnz == 0:
                candidates[s1_id] = {} if return_scores else set()
                continue

            # Sparse dot product for single query row against transposed target matrix
            row_sims = q_vec.dot(X_target_T)
            # Explicitly ensure CSR format for documented row-wise array extraction
            if not isinstance(row_sims, sp.csr_matrix):
                row_sims = row_sims.tocsr()

            if row_sims.nnz == 0:
                candidates[s1_id] = {} if return_scores else set()
                continue

            cols = row_sims.indices
            data = row_sims.data

            if min_similarity > 0.0:
                mask = data >= min_similarity
                cols = cols[mask]
                data = data[mask]
                if len(data) == 0:
                    candidates[s1_id] = {} if return_scores else set()
                    continue

            n_matches = len(data)
            if n_matches <= top_k:
                order = np.argsort(-data)
                cand_cols = cols[order]
                cand_scores = data[order]
            elif has_duplicate_targets:
                # If target IDs contain duplicates, sort all matches descending
                # so deduplication deterministically selects the top distinct entities
                order = np.argsort(-data)
                cand_cols = cols[order]
                cand_scores = data[order]
            else:
                # Top-K partition in O(M) time without full sort
                part = np.argpartition(-data, top_k)[:top_k]
                order = part[np.argsort(-data[part])]
                cand_cols = cols[order]
                cand_scores = data[order]

            selected_scores: Dict[str, float] = {}
            for cid_idx, score in zip(cand_cols, cand_scores):
                cid = str(target_ids[cid_idx])
                score_val = float(score)
                if cid not in selected_scores:
                    selected_scores[cid] = score_val
                    if len(selected_scores) >= top_k:
                        break
                elif score_val > selected_scores[cid]:
                    # Deterministically preserve the best (maximum) score if seen again
                    selected_scores[cid] = score_val

            if return_scores:
                candidates[s1_id] = selected_scores
            else:
                candidates[s1_id] = set(selected_scores.keys())

    return candidates


def retrieve_word_tfidf_candidates(
    s1_df: pd.DataFrame,
    target_df: Union[pd.DataFrame, List[pd.DataFrame]],
    top_k: int = WORD_TFIDF_TOP_K,
    ngram_range: Tuple[int, int] = WORD_NGRAM_RANGE,
    chunk_size: int = 500,
    min_similarity: float = 0.0,
    return_scores: bool = False,
    min_df: int = 1,
) -> Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """Retrieve candidate target IDs for each S1 entity using Word n-gram TF-IDF similarity.

    Memory-safe for multi-million-row target datasets:
    1. Uses word n-grams (default: (1, 2)) to capture whole-word token matches and word bigrams.
    2. Uses float32 TF-IDF matrix representation to cut memory by 50%.
    3. Keeps sparse TF-IDF matrices; never calls toarray() or todense().
    4. Retains chunked S1 query vectorization for efficient batch string transforms.
    5. Computes similarity row-by-row against precomputed transposed target matrix (X_target_T).
    6. Discards intermediate row similarities immediately after extracting Top-K candidates.
    7. Explicitly ensures CSR format for row-wise Top-K extraction.
    8. Handles duplicate target entity IDs by aggregating maximum similarity scores across
       all occurrences without premature stopping.
    9. Enforces deterministic tie-breaking using candidate entity_id as the secondary key.
    10. Handles missing, null, or empty business names gracefully.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 query DataFrame containing 'entity_id' and 'business_name' or 'name_normalized'.
    target_df : Union[pd.DataFrame, List[pd.DataFrame]]
        Target candidate DataFrame (or list of DataFrames for Source 2 and Source 3)
        containing 'entity_id' and 'business_name' or 'name_normalized'.
    top_k : int
        Maximum number of candidates to retrieve per S1 entity.
    ngram_range : Tuple[int, int]
        Word n-gram range for TF-IDF (defaults to (1, 2)).
    chunk_size : int
        Number of S1 query records processed per batch to bound memory usage.
    min_similarity : float
        Minimum cosine similarity threshold to consider a candidate.
    return_scores : bool
        If True, returns Dict[s1_id, Dict[cand_id, score]].
        If False, returns Dict[s1_id, Set[cand_id]].
    min_df : int
        Minimum document frequency for word n-grams. Defaults to 1.
        In entity resolution blocking, min_df=1 is strictly appropriate to preserve
        rare brand names, distinctive word tokens, and unique entity identifiers.

    Returns
    -------
    Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]]
        Mapping from source1_entity_id to candidate entity IDs (preserving S2- and S3- prefixes).
    """
    if "entity_id" not in s1_df.columns:
        raise ValueError("s1_df must contain 'entity_id' column.")

    # Combine target dataframes if provided as a list
    if isinstance(target_df, list):
        target_df = pd.concat(target_df, ignore_index=True)

    if "entity_id" not in target_df.columns:
        raise ValueError("target_df must contain 'entity_id' column.")

    target_ids = target_df["entity_id"].astype(str).tolist()
    target_names = _extract_normalized_names(target_df)

    n_targets = len(target_names)
    if n_targets == 0:
        return {str(sid): ({} if return_scores else set()) for sid in s1_df["entity_id"]}

    has_duplicate_targets = len(target_ids) != len(set(target_ids))

    # Fit Word TF-IDF on target candidate corpus with float32 dtype to halve memory usage
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        min_df=min_df,
        dtype=np.float32,
    )

    try:
        X_target = vectorizer.fit_transform(target_names)
    except ValueError as e:
        # Handle degenerate case where target documents contain no valid word tokens
        if "empty vocabulary" in str(e).lower():
            return {str(sid): ({} if return_scores else set()) for sid in s1_df["entity_id"]}
        raise

    if not isinstance(X_target, sp.csr_matrix):
        X_target = X_target.tocsr()

    # Precompute transpose once.
    X_target_T = X_target.T.tocsr()

    # Process S1 queries in chunks to bound memory
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    s1_names = _extract_normalized_names(s1_df)
    n_s1 = len(s1_ids)

    candidates: Dict[str, Union[Set[str], Dict[str, float]]] = {}

    for start_idx in range(0, n_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_s1)
        chunk_texts = s1_names[start_idx:end_idx]
        chunk_ids = s1_ids[start_idx:end_idx]

        # Chunked sparse query vectorization
        X_chunk = vectorizer.transform(chunk_texts)
        if not isinstance(X_chunk, sp.csr_matrix):
            X_chunk = X_chunk.tocsr()

        for i, s1_id in enumerate(chunk_ids):
            if top_k <= 0:
                candidates[s1_id] = {} if return_scores else set()
                continue

            q_vec = X_chunk[i : i + 1]  # 1 x D CSR vector

            if q_vec.nnz == 0:
                candidates[s1_id] = {} if return_scores else set()
                continue

            # Sparse dot product for single query row against transposed target matrix
            row_sims = q_vec.dot(X_target_T)
            # Explicitly ensure CSR format for documented row-wise array extraction
            if not isinstance(row_sims, sp.csr_matrix):
                row_sims = row_sims.tocsr()

            if row_sims.nnz == 0:
                candidates[s1_id] = {} if return_scores else set()
                continue

            cols = row_sims.indices
            data = row_sims.data

            if min_similarity > 0.0:
                mask = data >= min_similarity
                cols = cols[mask]
                data = data[mask]
                if len(data) == 0:
                    candidates[s1_id] = {} if return_scores else set()
                    continue

            # Aggregate scores if duplicate target IDs exist
            if has_duplicate_targets:
                entity_best_scores: Dict[str, float] = {}
                for col, score in zip(cols, data):
                    cid = target_ids[col]
                    s = float(score)
                    if cid not in entity_best_scores or s > entity_best_scores[cid]:
                        entity_best_scores[cid] = s
                items = list(entity_best_scores.items())
            else:
                items = [(target_ids[col], float(score)) for col, score in zip(cols, data)]

            # Deterministic selection: primary key -score (descending), secondary key cid (ascending)
            if len(items) <= top_k:
                best = sorted(items, key=lambda x: (-x[1], x[0]))
            else:
                best = heapq.nsmallest(top_k, items, key=lambda x: (-x[1], x[0]))

            if return_scores:
                candidates[s1_id] = {cid: score for cid, score in best}
            else:
                candidates[s1_id] = {cid for cid, _ in best}

    return candidates



def union_candidates(
    char_candidates: Dict[str, Set[str]],
    word_candidates: Optional[Dict[str, Set[str]]] = None,
    embedding_candidates: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, Set[str]]:
    """Form the Candidate UNION across Character TF-IDF, Word TF-IDF, and Embedding retrievals.

    Parameters
    ----------
    char_candidates : Dict[str, Set[str]]
        Candidates from Character TF-IDF branch.
    word_candidates : Optional[Dict[str, Set[str]]]
        Candidates from Word TF-IDF branch (optional, defaults to empty).
    embedding_candidates : Optional[Dict[str, Set[str]]]
        Candidates from Embedding retrieval branch (optional, defaults to empty).

    Returns
    -------
    Dict[str, Set[str]]
        Mapping of source1_entity_id -> combined candidate entity_ids.
    """
    word_candidates = word_candidates or {}
    embedding_candidates = embedding_candidates or {}

    all_s1_ids = set(char_candidates) | set(word_candidates) | set(embedding_candidates)
    combined: Dict[str, Set[str]] = {}
    for s1_id in all_s1_ids:
        c_set = set()
        c_set.update(char_candidates.get(s1_id, set()))
        c_set.update(word_candidates.get(s1_id, set()))
        c_set.update(embedding_candidates.get(s1_id, set()))
        combined[s1_id] = c_set
    return combined


def evaluate_blocking_recall(
    candidates: Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]],
    ground_truth_path: Union[str, Path],
    s1_ids_subset: Optional[Set[str]] = None,
) -> Dict[str, Union[float, int]]:
    """Evaluate candidate pairs against ground truth matching labels.

    Parameters
    ----------
    candidates : Union[Dict[str, Set[str]], Dict[str, Dict[str, float]]]
        Mapping of source1_entity_id -> set of candidate entity_ids (or dict of candidate -> score).
    ground_truth_path : Union[str, Path]
        Path to train_ground_truth.tsv.
    s1_ids_subset : Optional[Set[str]]
        Optional subset of S1 entity IDs to evaluate over. If None, evaluates
        over all S1 entity IDs present in `candidates`.

    Returns
    -------
    Dict[str, Union[float, int]]
        Metrics:
        - 'pair_recall': retrieved_true_pairs / total_true_pairs
        - 'average_candidates_per_s1': mean candidate count per S1 entity
        - 'number_of_s1_evaluated': total S1 entities evaluated
        - 'total_true_pairs': total true matching pairs in ground truth
        - 'retrieved_true_pairs': true matching pairs captured by candidates
    """
    eval_s1_ids = set(candidates.keys()) if s1_ids_subset is None else (set(candidates.keys()) & set(s1_ids_subset))
    if not eval_s1_ids:
        return {
            "pair_recall": 0.0,
            "average_candidates_per_s1": 0.0,
            "number_of_s1_evaluated": 0,
            "total_true_pairs": 0,
            "retrieved_true_pairs": 0,
        }

    total_true_pairs = 0
    retrieved_true_pairs = 0
    evaluated_with_gt: Set[str] = set()

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            s1_id = parts[0]
            if s1_id in eval_s1_ids:
                evaluated_with_gt.add(s1_id)
                matched_str = parts[1] if len(parts) > 1 else ""
                if not matched_str:
                    continue
                matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                total_true_pairs += len(matched_ids)

                cand_entry = candidates.get(s1_id, set())
                cand_set = set(cand_entry.keys()) if isinstance(cand_entry, dict) else set(cand_entry)
                retrieved_true_pairs += sum(1 for m in matched_ids if m in cand_set)

    # Detect entity ID mismatch where zero candidate S1 IDs matched any record in ground truth
    if len(evaluated_with_gt) == 0:
        raise ValueError(
            f"Evaluation failed: None of the {len(eval_s1_ids)} candidate S1 entity IDs joined with "
            f"ground truth at '{ground_truth_path}'. Check for entity ID format mismatch or incorrect ground truth file."
        )

    pair_recall = (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    total_candidates = sum(len(candidates.get(sid, set())) for sid in eval_s1_ids)
    avg_candidates = total_candidates / len(eval_s1_ids) if eval_s1_ids else 0.0

    return {
        "pair_recall": float(pair_recall),
        "average_candidates_per_s1": float(avg_candidates),
        "number_of_s1_evaluated": len(eval_s1_ids),
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
    }
