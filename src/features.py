"""Feature engineering module for candidate pairs in Business Entity Resolution.

Locked Architecture:
S1 -> (Char TF-IDF, Word TF-IDF, Embedding) -> Candidate Union -> ALL Features -> LightGBM -> F0.5

Every candidate pair that survives the candidate union receives the complete set of
pairwise similarity features:
- Normalized name exact-match indicator
- Normalized compact name exact-match indicator
- Name character TF-IDF / n-gram similarity
- Name word TF-IDF / n-gram similarity
- Name token Jaccard & Dice overlap
- Name length difference and ratio
- Address normalized exact-match indicator
- Address compact exact-match indicator
- Address string similarity
- Address token overlap
- Address numeric token overlap and exact-match
- Address length difference
- Country exact-match, mismatch, and missing flags
- Missingness indicators for names and addresses
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

# Sentinel empty dict for safe .get() without allocation
_EMPTY_DICT: Dict = {}

from src.preprocess import (
    compact_address,
    compact_name,
    extract_numeric_tokens,
    normalize_address,
    normalize_name,
)

# Standard list of numerical feature columns used for model training and scoring
FEATURE_COLUMNS = [
    "name_exact_match",
    "name_compact_exact_match",
    "name_char_tfidf_sim",
    "name_word_tfidf_sim",
    "name_token_overlap",
    "name_token_dice",
    "name_len_diff",
    "name_len_ratio",
    "name_has_missing",
    "address_exact_match",
    "address_compact_exact_match",
    "address_string_sim",
    "address_token_overlap",
    "address_numeric_overlap",
    "address_numeric_exact_match",
    "address_numeric_mismatch",
    "address_len_diff",
    "address_has_missing",
    "country_exact_match",
    "country_mismatch",
    "country_has_missing",
    "emb_sim",
]


def _char_ngram_similarity(s1: str, s2: str, n_min: int = 3, n_max: int = 5) -> float:
    """Compute character n-gram cosine similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    l1, l2 = len(s1), len(s2)
    if l1 < n_min or l2 < n_min:
        return 1.0 if s1 == s2 else 0.0

    ng1 = Counter(s1[i : i + n] for n in range(n_min, n_max + 1) for i in range(l1 - n + 1))
    ng2 = Counter(s2[i : i + n] for n in range(n_min, n_max + 1) for i in range(l2 - n + 1))
    if not ng1 or not ng2:
        return 0.0

    common = set(ng1) & set(ng2)
    if not common:
        return 0.0

    dot = sum(ng1[t] * ng2[t] for t in common)
    norm1 = math.sqrt(sum(v * v for v in ng1.values()))
    norm2 = math.sqrt(sum(v * v for v in ng2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return min(1.0, max(0.0, float(dot / (norm1 * norm2))))


def _word_ngram_similarity(s1: str, s2: str) -> float:
    """Compute word (1, 2) n-gram cosine similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    toks1 = s1.split()
    toks2 = s2.split()
    if not toks1 or not toks2:
        return 0.0

    ng1 = Counter(toks1)
    ng1.update(" ".join(toks1[i : i + 2]) for i in range(len(toks1) - 1))
    ng2 = Counter(toks2)
    ng2.update(" ".join(toks2[i : i + 2]) for i in range(len(toks2) - 1))

    common = set(ng1) & set(ng2)
    if not common:
        return 0.0

    dot = sum(ng1[t] * ng2[t] for t in common)
    norm1 = math.sqrt(sum(v * v for v in ng1.values()))
    norm2 = math.sqrt(sum(v * v for v in ng2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return min(1.0, max(0.0, float(dot / (norm1 * norm2))))


def _token_jaccard(toks1: List[str], toks2: List[str]) -> float:
    """Compute Jaccard token overlap between two token lists."""
    if not toks1 or not toks2:
        return 0.0
    s1 = set(toks1)
    s2 = set(toks2)
    union_len = len(s1 | s2)
    if union_len == 0:
        return 0.0
    return min(1.0, max(0.0, float(len(s1 & s2) / union_len)))


def _token_dice(toks1: List[str], toks2: List[str]) -> float:
    """Compute Dice token similarity between two token lists."""
    if not toks1 or not toks2:
        return 0.0
    s1 = set(toks1)
    s2 = set(toks2)
    denom = len(s1) + len(s2)
    if denom == 0:
        return 0.0
    return min(1.0, max(0.0, float(2.0 * len(s1 & s2) / denom)))


def _numeric_token_overlap(nums1: List[str], nums2: List[str]) -> Tuple[float, float, float]:
    """Compute numeric token overlap metrics.

    Returns
    -------
    Tuple[float, float, float]
        (numeric_jaccard_overlap, numeric_exact_match, numeric_mismatch)
    """
    if not nums1 or not nums2:
        return 0.0, 0.0, 0.0
    set1 = set(nums1)
    set2 = set(nums2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = float(inter / union) if union > 0 else 0.0
    exact = 1.0 if set1 == set2 else 0.0
    mismatch = 1.0 if inter == 0 else 0.0
    return jaccard, exact, mismatch


def compute_string_similarities(s1_text: str, s2_text: str) -> dict:
    """Compute string-level similarity metrics between two text records.

    Parameters
    ----------
    s1_text : str
        First text string (e.g. name or address).
    s2_text : str
        Second text string (e.g. candidate name or address).

    Returns
    -------
    dict
        Dictionary of string similarity metrics:
        - exact_match: 1.0 if identical non-empty, 0.0 otherwise
        - char_ngram_sim: character (3, 5) n-gram cosine similarity
        - word_ngram_sim: word (1, 2) n-gram cosine similarity
        - token_overlap: token Jaccard similarity
        - token_dice: token Dice similarity
        - len_diff: absolute length difference
        - len_ratio: length ratio min(l1, l2) / max(l1, l2)
    """
    s1 = (s1_text or "").strip()
    s2 = (s2_text or "").strip()

    if not s1 or not s2:
        return {
            "exact_match": 0.0,
            "char_ngram_sim": 0.0,
            "word_ngram_sim": 0.0,
            "token_overlap": 0.0,
            "token_dice": 0.0,
            "len_diff": float(abs(len(s1) - len(s2))),
            "len_ratio": 0.0,
        }

    exact = 1.0 if s1 == s2 else 0.0
    l1, l2 = len(s1), len(s2)
    len_diff = float(abs(l1 - l2))
    len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    toks1 = s1.split()
    toks2 = s2.split()
    token_overlap = _token_jaccard(toks1, toks2)
    token_dice = _token_dice(toks1, toks2)

    char_sim = _char_ngram_similarity(s1, s2)
    word_sim = _word_ngram_similarity(s1, s2)

    return {
        "exact_match": exact,
        "char_ngram_sim": char_sim,
        "word_ngram_sim": word_sim,
        "token_overlap": token_overlap,
        "token_dice": token_dice,
        "len_diff": len_diff,
        "len_ratio": len_ratio,
    }


def extract_pair_features(
    s1_record: dict,
    candidate_record: dict,
    char_sim: float = 0.0,
    word_sim: float = 0.0,
    emb_sim: float = 0.0,
) -> dict:
    """Compute the full set of similarity features for a candidate pair.

    Every candidate pair surviving the candidate union receives all similarity features:
    - Name similarities (exact match, compact exact match, char TF-IDF, word TF-IDF, token overlaps, length diff)
    - Address similarities (exact match, compact exact match, string similarity, token overlap, numeric overlap, length diff)
    - Country compatibility flags (exact match, mismatch, missing)
    - Missingness indicators for names and addresses

    Parameters
    ----------
    s1_record : dict
        Source 1 record containing business attributes ('business_name', 'address', 'country').
    candidate_record : dict
        Candidate record (Source 2 or Source 3) containing business attributes.
    char_sim : float
        Optional precomputed Character TF-IDF retrieval score from blocking.
    word_sim : float
        Optional precomputed Word TF-IDF retrieval score from blocking.
    emb_sim : float
        Optional precomputed Embedding cosine similarity (defaults to 0.0).

    Returns
    -------
    dict
        Feature vector mapping feature names to strictly finite numerical floats.
    """
    # 1. Extract and normalize business names
    name1 = s1_record.get("name_normalized")
    if name1 is None or pd.isna(name1) or name1 == "":
        name1 = normalize_name(s1_record.get("business_name"))

    name2 = candidate_record.get("name_normalized")
    if name2 is None or pd.isna(name2) or name2 == "":
        name2 = normalize_name(candidate_record.get("business_name"))

    compact1 = s1_record.get("name_compact")
    if compact1 is None or pd.isna(compact1) or compact1 == "":
        compact1 = compact_name(s1_record.get("business_name"))

    compact2 = candidate_record.get("name_compact")
    if compact2 is None or pd.isna(compact2) or compact2 == "":
        compact2 = compact_name(candidate_record.get("business_name"))

    # 2. Extract and normalize addresses
    addr1 = s1_record.get("address_normalized")
    if addr1 is None or pd.isna(addr1) or addr1 == "":
        addr1 = normalize_address(s1_record.get("address"))

    addr2 = candidate_record.get("address_normalized")
    if addr2 is None or pd.isna(addr2) or addr2 == "":
        addr2 = normalize_address(candidate_record.get("address"))

    addr_compact1 = s1_record.get("address_compact")
    if addr_compact1 is None or pd.isna(addr_compact1) or addr_compact1 == "":
        addr_compact1 = compact_address(s1_record.get("address"))

    addr_compact2 = candidate_record.get("address_compact")
    if addr_compact2 is None or pd.isna(addr_compact2) or addr_compact2 == "":
        addr_compact2 = compact_address(candidate_record.get("address"))

    # 3. Extract and normalize countries
    raw_c1 = s1_record.get("country_normalized") or s1_record.get("country") or ""
    c1 = str(raw_c1).strip().lower()
    if c1 in ("nan", "none", "null", "<null>"):
        c1 = ""

    raw_c2 = candidate_record.get("country_normalized") or candidate_record.get("country") or ""
    c2 = str(raw_c2).strip().lower()
    if c2 in ("nan", "none", "null", "<null>"):
        c2 = ""

    # 4. Extract numeric tokens from addresses
    nums1 = s1_record.get("address_numeric_tokens")
    if nums1 is None or not isinstance(nums1, list):
        nums1 = extract_numeric_tokens(s1_record.get("address"))

    nums2 = candidate_record.get("address_numeric_tokens")
    if nums2 is None or not isinstance(nums2, list):
        nums2 = extract_numeric_tokens(candidate_record.get("address"))

    # ---------------------------------------------------------
    # Name Features
    # ---------------------------------------------------------
    has_name1 = bool(name1)
    has_name2 = bool(name2)
    name_has_missing = 1.0 if not (has_name1 and has_name2) else 0.0

    name_exact = 1.0 if (has_name1 and has_name2 and name1 == name2) else 0.0
    name_compact_exact = 1.0 if (compact1 and compact2 and compact1 == compact2) else 0.0

    # Use precomputed retrieval score if positive; otherwise compute n-gram similarity
    if char_sim > 0.0:
        name_char_sim = float(char_sim)
    else:
        name_char_sim = 1.0 if name_exact else _char_ngram_similarity(name1, name2)

    if word_sim > 0.0:
        name_word_sim = float(word_sim)
    else:
        name_word_sim = 1.0 if name_exact else _word_ngram_similarity(name1, name2)

    name_toks1 = name1.split()
    name_toks2 = name2.split()
    name_token_overlap = _token_jaccard(name_toks1, name_toks2)
    name_token_dice = _token_dice(name_toks1, name_toks2)

    l_name1, l_name2 = len(name1), len(name2)
    name_len_diff = float(abs(l_name1 - l_name2))
    name_len_ratio = float(min(l_name1, l_name2) / max(l_name1, l_name2)) if max(l_name1, l_name2) > 0 else 0.0

    # ---------------------------------------------------------
    # Address Features
    # ---------------------------------------------------------
    has_addr1 = bool(addr1)
    has_addr2 = bool(addr2)
    addr_has_missing = 1.0 if not (has_addr1 and has_addr2) else 0.0

    addr_exact = 1.0 if (has_addr1 and has_addr2 and addr1 == addr2) else 0.0
    addr_compact_exact = 1.0 if (addr_compact1 and addr_compact2 and addr_compact1 == addr_compact2) else 0.0

    addr_string_sim = 1.0 if addr_exact else _char_ngram_similarity(addr1, addr2)
    addr_toks1 = addr1.split()
    addr_toks2 = addr2.split()
    addr_token_overlap = _token_jaccard(addr_toks1, addr_toks2)

    num_overlap, num_exact, num_mismatch = _numeric_token_overlap(nums1, nums2)

    l_addr1, l_addr2 = len(addr1), len(addr2)
    addr_len_diff = float(abs(l_addr1 - l_addr2))

    # ---------------------------------------------------------
    # Country Features
    # ---------------------------------------------------------
    if c1 and c2:
        country_exact = 1.0 if c1 == c2 else 0.0
        country_mismatch = 1.0 if c1 != c2 else 0.0
        country_has_missing = 0.0
    else:
        country_exact = 0.0
        country_mismatch = 0.0
        country_has_missing = 1.0

    return {
        "name_exact_match": name_exact,
        "name_compact_exact_match": name_compact_exact,
        "name_char_tfidf_sim": name_char_sim,
        "name_word_tfidf_sim": name_word_sim,
        "name_token_overlap": name_token_overlap,
        "name_token_dice": name_token_dice,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_has_missing": name_has_missing,
        "address_exact_match": addr_exact,
        "address_compact_exact_match": addr_compact_exact,
        "address_string_sim": addr_string_sim,
        "address_token_overlap": addr_token_overlap,
        "address_numeric_overlap": num_overlap,
        "address_numeric_exact_match": num_exact,
        "address_numeric_mismatch": num_mismatch,
        "address_len_diff": addr_len_diff,
        "address_has_missing": addr_has_missing,
        "country_exact_match": country_exact,
        "country_mismatch": country_mismatch,
        "country_has_missing": country_has_missing,
        "emb_sim": float(emb_sim),
    }


def build_feature_matrix(
    candidate_pairs: Union[List[Tuple[str, str]], pd.DataFrame],
    s1_lookup: Union[pd.DataFrame, Dict[str, dict]],
    target_lookup: Union[pd.DataFrame, Dict[str, dict]],
    char_scores: Optional[Dict[str, Dict[str, float]]] = None,
    word_scores: Optional[Dict[str, Dict[str, float]]] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for all candidate pairs surviving the union.

    Parameters
    ----------
    candidate_pairs : Union[List[Tuple[str, str]], pd.DataFrame]
        Candidate pairs as a list of (s1_id, candidate_id) tuples or a DataFrame
        containing identifier columns (e.g. ['source1_entity_id', 'matched_entity_id']).
    s1_lookup : Union[pd.DataFrame, Dict[str, dict]]
        Source 1 records indexed by entity_id or as a mapping {entity_id: record_dict}.
    target_lookup : Union[pd.DataFrame, Dict[str, dict]]
        Target candidate records (Source 2 / 3) indexed by entity_id or as a mapping.
    char_scores : Optional[Dict[str, Dict[str, float]]]
        Optional precomputed Character TF-IDF retrieval scores {s1_id: {cand_id: score}}.
    word_scores : Optional[Dict[str, Dict[str, float]]]
        Optional precomputed Word TF-IDF retrieval scores {s1_id: {cand_id: score}}.
    chunk_size : Optional[int]
        Optional batch size for memory-conscious candidate pair processing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing pair identifiers and all computed numerical features,
        with strictly finite values and zero NaN/inf entries.
    """
    # Normalize lookup structures to fast-access mappings if DataFrames
    if isinstance(s1_lookup, pd.DataFrame):
        if "entity_id" in s1_lookup.columns:
            s1_dict = s1_lookup.set_index("entity_id").to_dict(orient="index")
        else:
            s1_dict = s1_lookup.to_dict(orient="index")
    else:
        s1_dict = s1_lookup

    if isinstance(target_lookup, pd.DataFrame):
        if "entity_id" in target_lookup.columns:
            target_dict = target_lookup.set_index("entity_id").to_dict(orient="index")
        else:
            target_dict = target_lookup.to_dict(orient="index")
    else:
        target_dict = target_lookup

    # Normalize candidate pairs to a list of (s1_id, cand_id, char_sim, word_sim) tuples
    pairs_list: List[Tuple[str, str, float, float]] = []

    if isinstance(candidate_pairs, pd.DataFrame):
        cols = candidate_pairs.columns
        if "source1_entity_id" in cols and "matched_entity_id" in cols:
            s1_col = "source1_entity_id"
            cand_col = "matched_entity_id"
        elif "s1_id" in cols and "target_id" in cols:
            s1_col = "s1_id"
            cand_col = "target_id"
        elif "s1_id" in cols and "candidate_id" in cols:
            s1_col = "s1_id"
            cand_col = "candidate_id"
        elif len(cols) >= 2:
            s1_col = cols[0]
            cand_col = cols[1]
        else:
            raise ValueError("candidate_pairs DataFrame must have at least 2 columns")

        char_col = "char_sim" if "char_sim" in cols else ("char_score" if "char_score" in cols else None)
        word_col = "word_sim" if "word_sim" in cols else ("word_score" if "word_score" in cols else None)

        for _, row in candidate_pairs.iterrows():
            sid = str(row[s1_col])
            cid = str(row[cand_col])
            csim = float(row[char_col]) if char_col and pd.notna(row[char_col]) else 0.0
            wsim = float(row[word_col]) if word_col and pd.notna(row[word_col]) else 0.0
            pairs_list.append((sid, cid, csim, wsim))
    else:
        for item in candidate_pairs:
            sid = str(item[0])
            cid = str(item[1])
            csim = float(item[2]) if len(item) > 2 else 0.0
            wsim = float(item[3]) if len(item) > 3 else 0.0
            pairs_list.append((sid, cid, csim, wsim))

    if not pairs_list:
        empty_cols = ["source1_entity_id", "matched_entity_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=empty_cols)

    # Process pairs in batches if chunk_size is provided, or in a single pass
    total_pairs = len(pairs_list)
    step = chunk_size if chunk_size and chunk_size > 0 else total_pairs
    rows = []

    for start_idx in range(0, total_pairs, step):
        batch = pairs_list[start_idx : start_idx + step]
        for sid, cid, csim, wsim in batch:
            s1_rec = s1_dict.get(sid, {})
            cand_rec = target_dict.get(cid, {})

            # Lookup precomputed scores from mappings if available
            if csim == 0.0 and char_scores:
                csim = float(char_scores.get(sid, {}).get(cid, 0.0))
            if wsim == 0.0 and word_scores:
                wsim = float(word_scores.get(sid, {}).get(cid, 0.0))

            feats = extract_pair_features(
                s1_rec,
                cand_rec,
                char_sim=csim,
                word_sim=wsim,
            )
            feats["source1_entity_id"] = sid
            feats["matched_entity_id"] = cid
            rows.append(feats)

    df = pd.DataFrame(rows)

    # Ensure identifier columns are placed first followed by standard feature order
    id_cols = ["source1_entity_id", "matched_entity_id"]
    ordered_cols = id_cols + [c for c in FEATURE_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    # Enforce float32 for all numerical feature columns to bound memory
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    return df


def build_feature_matrix_v2(
    s1_ids_list: List[str],
    cand_ids_list: List[str],
    char_sims: np.ndarray,
    word_sims: np.ndarray,
    s1_dict: Dict[str, dict],
    target_dict: Dict[str, dict],
) -> pd.DataFrame:
    """Vectorized feature matrix construction using pre-normalised attribute dicts.

    Semantics are identical to build_feature_matrix / extract_pair_features but
    avoids per-pair Python overhead:
    - No per-pair dict creation or pd.isna() calls.
    - Two attribute-extraction passes over N pairs (one for S1, one for targets).
    - DataFrame built from column arrays (not from list-of-row-dicts).

    Requires s1_dict and target_dict to carry pre-normalised keys:
        name_normalized, name_compact,
        address_normalized, address_compact,
        address_numeric_tokens  (list[str] or []),
        country_norm_lc         (casefolded str, '' if missing)

    Falls back gracefully to empty strings / empty lists for any missing key.
    """
    N = len(s1_ids_list)
    if N == 0:
        empty_cols = ["source1_entity_id", "matched_entity_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=empty_cols)

    # ── Attribute extraction (two list-level passes, no pd.isna) ─────────────
    n1: List[str] = [''] * N
    nc1: List[str] = [''] * N
    a1: List[str] = [''] * N
    ac1: List[str] = [''] * N
    nums1: List[list] = [_EMPTY_DICT] * N  # reused sentinel; replaced below
    c1: List[str] = [''] * N

    n2: List[str] = [''] * N
    nc2: List[str] = [''] * N
    a2: List[str] = [''] * N
    ac2: List[str] = [''] * N
    nums2: List[list] = [_EMPTY_DICT] * N
    c2: List[str] = [''] * N

    for i in range(N):
        sr = s1_dict.get(s1_ids_list[i], _EMPTY_DICT)
        tr = target_dict.get(cand_ids_list[i], _EMPTY_DICT)

        n1[i] = sr.get('name_normalized') or ''
        nc1[i] = sr.get('name_compact') or ''
        a1[i] = sr.get('address_normalized') or ''
        ac1[i] = sr.get('address_compact') or ''
        nums1[i] = sr.get('address_numeric_tokens') or []
        c1[i] = sr.get('country_norm_lc') or ''

        n2[i] = tr.get('name_normalized') or ''
        nc2[i] = tr.get('name_compact') or ''
        a2[i] = tr.get('address_normalized') or ''
        ac2[i] = tr.get('address_compact') or ''
        nums2[i] = tr.get('address_numeric_tokens') or []
        c2[i] = tr.get('country_norm_lc') or ''

    # ── Name features ─────────────────────────────────────────────────────────
    name_exact = np.empty(N, dtype=np.float32)
    for i in range(N):
        name_exact[i] = 1.0 if (n1[i] and n2[i] and n1[i] == n2[i]) else 0.0

    name_compact_exact = np.empty(N, dtype=np.float32)
    for i in range(N):
        name_compact_exact[i] = 1.0 if (nc1[i] and nc2[i] and nc1[i] == nc2[i]) else 0.0

    # Use precomputed retrieval score when > 0; otherwise compute n-gram similarity
    cs = np.asarray(char_sims, dtype=np.float32).copy()
    ws = np.asarray(word_sims, dtype=np.float32).copy()
    for i in range(N):
        if cs[i] == 0.0:
            cs[i] = 1.0 if name_exact[i] else _char_ngram_similarity(n1[i], n2[i])
        if ws[i] == 0.0:
            ws[i] = 1.0 if name_exact[i] else _word_ngram_similarity(n1[i], n2[i])

    name_toks1 = [x.split() for x in n1]
    name_toks2 = [x.split() for x in n2]
    name_token_overlap = np.empty(N, dtype=np.float32)
    name_token_dice_arr = np.empty(N, dtype=np.float32)
    for i in range(N):
        name_token_overlap[i] = _token_jaccard(name_toks1[i], name_toks2[i])
        name_token_dice_arr[i] = _token_dice(name_toks1[i], name_toks2[i])

    l_n1 = np.array([len(x) for x in n1], dtype=np.float32)
    l_n2 = np.array([len(x) for x in n2], dtype=np.float32)
    name_len_diff = np.abs(l_n1 - l_n2)
    mx_n = np.maximum(l_n1, l_n2)
    name_len_ratio = np.where(mx_n > 0, np.minimum(l_n1, l_n2) / mx_n, 0.0).astype(np.float32)
    name_has_missing = np.empty(N, dtype=np.float32)
    for i in range(N):
        name_has_missing[i] = 0.0 if (n1[i] and n2[i]) else 1.0

    # ── Address features ──────────────────────────────────────────────────────
    addr_exact = np.empty(N, dtype=np.float32)
    for i in range(N):
        addr_exact[i] = 1.0 if (a1[i] and a2[i] and a1[i] == a2[i]) else 0.0

    addr_compact_exact = np.empty(N, dtype=np.float32)
    for i in range(N):
        addr_compact_exact[i] = 1.0 if (ac1[i] and ac2[i] and ac1[i] == ac2[i]) else 0.0

    addr_string_sim = np.empty(N, dtype=np.float32)
    for i in range(N):
        addr_string_sim[i] = 1.0 if addr_exact[i] else _char_ngram_similarity(a1[i], a2[i])

    addr_toks1 = [x.split() for x in a1]
    addr_toks2 = [x.split() for x in a2]
    addr_token_overlap = np.empty(N, dtype=np.float32)
    for i in range(N):
        addr_token_overlap[i] = _token_jaccard(addr_toks1[i], addr_toks2[i])

    addr_numeric_overlap = np.empty(N, dtype=np.float32)
    addr_numeric_exact = np.empty(N, dtype=np.float32)
    addr_numeric_mismatch = np.empty(N, dtype=np.float32)
    for i in range(N):
        jac, exc, mis = _numeric_token_overlap(nums1[i], nums2[i])
        addr_numeric_overlap[i] = jac
        addr_numeric_exact[i] = exc
        addr_numeric_mismatch[i] = mis

    l_a1 = np.array([len(x) for x in a1], dtype=np.float32)
    l_a2 = np.array([len(x) for x in a2], dtype=np.float32)
    addr_len_diff = np.abs(l_a1 - l_a2)
    addr_has_missing = np.empty(N, dtype=np.float32)
    for i in range(N):
        addr_has_missing[i] = 0.0 if (a1[i] and a2[i]) else 1.0

    # ── Country features ──────────────────────────────────────────────────────
    country_exact = np.zeros(N, dtype=np.float32)
    country_mismatch = np.zeros(N, dtype=np.float32)
    country_has_missing = np.ones(N, dtype=np.float32)
    for i in range(N):
        if c1[i] and c2[i]:
            country_has_missing[i] = 0.0
            if c1[i] == c2[i]:
                country_exact[i] = 1.0
            else:
                country_mismatch[i] = 1.0

    # ── Build DataFrame from column arrays (no row-dict intermediate) ─────────
    df = pd.DataFrame({
        "source1_entity_id": s1_ids_list,
        "matched_entity_id": cand_ids_list,
        "name_exact_match": name_exact,
        "name_compact_exact_match": name_compact_exact,
        "name_char_tfidf_sim": cs,
        "name_word_tfidf_sim": ws,
        "name_token_overlap": name_token_overlap,
        "name_token_dice": name_token_dice_arr,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_has_missing": name_has_missing,
        "address_exact_match": addr_exact,
        "address_compact_exact_match": addr_compact_exact,
        "address_string_sim": addr_string_sim,
        "address_token_overlap": addr_token_overlap,
        "address_numeric_overlap": addr_numeric_overlap,
        "address_numeric_exact_match": addr_numeric_exact,
        "address_numeric_mismatch": addr_numeric_mismatch,
        "address_len_diff": addr_len_diff,
        "address_has_missing": addr_has_missing,
        "country_exact_match": country_exact,
        "country_mismatch": country_mismatch,
        "country_has_missing": country_has_missing,
        "emb_sim": np.zeros(N, dtype=np.float32),
    })

    return df
