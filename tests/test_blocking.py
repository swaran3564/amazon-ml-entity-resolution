"""Unit tests for the character TF-IDF candidate generation branch and blocking recall evaluation."""

import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking import (
    _extract_normalized_names,
    retrieve_char_tfidf_candidates,
    retrieve_word_tfidf_candidates,
    union_candidates,
    evaluate_blocking_recall,
)


# ============================================================================
# 1. Typo Variation Retrieval
# ============================================================================

def test_typo_variation_retrieval():
    """Verify character n-grams bridge typos and spelling variations observed in the dataset."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": [
            "Warwick Hills Golf Club",
            "Velasquez Ragland and Simmons Inc",
            "Obsidian LLC",
        ]
    })

    target_df = pd.DataFrame({
        "entity_id": ["S2-101", "S3-202", "S2-303", "S3-404"],
        "business_name": [
            "Warwik Hills Golf Club",                # internal character typo: ck -> k
            "VELASQUEZ, RAGLAND & SIMMON INC",       # dropped 's', uppercase, punctuation
            "Obsidian, [[LLC]]",                    # bracket noise
            "Zeta Zenith Bakery",                   # negative control with 0 overlap
        ]
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5)

    assert "S2-101" in candidates["S1-001"]
    assert "S3-202" in candidates["S1-002"]
    assert "S2-303" in candidates["S1-003"]
    assert "S3-404" not in candidates["S1-003"]


# ============================================================================
# 2. Top-K Limit Respected
# ============================================================================

def test_top_k_limit_respected():
    """Verify that candidate generation respects the configured top-K cutoff."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Acme Global Logistics"],
    })

    # Create 30 target entities with similar names
    target_records = [
        {"entity_id": f"S2-{i:03d}", "business_name": f"Acme Global Logistics Branch {i}"}
        for i in range(30)
    ]
    target_df = pd.DataFrame(target_records)

    top_k = 5
    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=top_k)

    assert len(candidates["S1-001"]) == top_k


# ============================================================================
# 3. Duplicate Candidate IDs Removed
# ============================================================================

def test_duplicate_candidate_ids_removed():
    """Verify candidate sets contain strictly unique entity IDs per S1 entity."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Style Multi Private Limited"],
    })

    target_df = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002", "S3-003"],
        "business_name": [
            "Style Multi Pvt Ltd",
            "Style Multi Private Limited",
            "Style Multi",
        ]
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=10)
    cand_set = candidates["S1-001"]

    assert isinstance(cand_set, set)
    # Check that candidate count matches unique count
    assert len(cand_set) == len(set(cand_set))
    assert {"S2-001", "S2-002", "S3-003"}.issubset(cand_set)


# ============================================================================
# 4. Source IDs Preserved
# ============================================================================

def test_source_ids_preserved():
    """Verify S2- and S3- entity ID prefixes are retained without cross-source confusion."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Summit Health Care"],
    })

    # S2 and S3 with potentially overlapping numeric suffixes
    s2_df = pd.DataFrame({
        "entity_id": ["S2-999"],
        "business_name": ["Summit Health Care Group"],
    })
    s3_df = pd.DataFrame({
        "entity_id": ["S3-999"],
        "business_name": ["Summit Health Care Clinic"],
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, [s2_df, s3_df], top_k=5)

    assert "S2-999" in candidates["S1-001"]
    assert "S3-999" in candidates["S1-001"]
    assert len(candidates["S1-001"]) == 2


# ============================================================================
# 5. No Dense NxM Matrix Created
# ============================================================================

def test_no_dense_nxm_matrix_created():
    """Ensure no full dense N x M similarity matrix is constructed during retrieval."""
    s1_df = pd.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(10)],
        "business_name": [f"Business Name {i}" for i in range(10)],
    })
    target_df = pd.DataFrame({
        "entity_id": [f"S2-{j}" for j in range(50)],
        "business_name": [f"Target Name {j}" for j in range(50)],
    })

    # Patch dense conversion methods to confirm they are never invoked
    with patch.object(sp.csr_matrix, "toarray", side_effect=AssertionError("Dense toarray() called!")):
        with patch.object(sp.csr_matrix, "todense", side_effect=AssertionError("Dense todense() called!")):
            candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5, chunk_size=5)

    assert len(candidates) == 10


# ============================================================================
# 6. Chunking Consistency
# ============================================================================

def test_chunking_consistency():
    """Verify that different chunk sizes yield identical candidate sets."""
    s1_df = pd.DataFrame({
        "entity_id": [f"S1-{i:02d}" for i in range(12)],
        "business_name": [f"Sample Enterprise {i}" for i in range(12)],
    })
    target_df = pd.DataFrame({
        "entity_id": [f"S2-{j:02d}" for j in range(20)],
        "business_name": [f"Sample Enterprise {j}" for j in range(20)],
    })

    cands_chunk1 = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=3, chunk_size=1)
    cands_chunk5 = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=3, chunk_size=5)
    cands_chunk12 = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=3, chunk_size=12)

    assert cands_chunk1 == cands_chunk5 == cands_chunk12


# ============================================================================
# 7. Candidate Union
# ============================================================================

def test_union_candidates():
    """Verify candidate union combines sets across branches without data loss."""
    char_cands = {"S1-001": {"S2-101", "S3-102"}, "S1-002": {"S2-201"}}
    word_cands = {"S1-001": {"S2-101", "S2-103"}, "S1-002": {"S3-202"}}
    emb_cands = {"S1-001": {"S3-104"}}

    combined = union_candidates(char_cands, word_cands, emb_cands)

    assert combined["S1-001"] == {"S2-101", "S3-102", "S2-103", "S3-104"}
    assert combined["S1-002"] == {"S2-201", "S3-202"}


# ============================================================================
# 8. Blocking Recall Evaluation Helper
# ============================================================================

def test_evaluate_blocking_recall_helper(tmp_path: Path):
    """Verify blocking recall evaluation helper with known synthetic ground truth."""
    gt_file = tmp_path / "synthetic_ground_truth.tsv"
    # Ground truth:
    # S1-001 matches S2-01, S3-01 (2 matches)
    # S1-002 matches S2-02 (1 match)
    # S1-003 has no matches (singleton, 0 matches)
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-01,S3-01\n"
        "S1-002\tS2-02\n"
        "S1-003\t\n"
    )
    gt_file.write_text(gt_content, encoding="utf-8")

    # Candidates:
    # S1-001 has S2-01, S2-99 (found 1 of 2 matches)
    # S1-002 has S2-02, S3-88 (found 1 of 1 match)
    # S1-003 has S2-77 (false candidate for singleton)
    candidates = {
        "S1-001": {"S2-01", "S2-99"},
        "S1-002": {"S2-02", "S3-88"},
        "S1-003": {"S2-77"},
    }

    metrics = evaluate_blocking_recall(candidates, gt_file)

    # Total true pairs: 2 (from S1-001) + 1 (from S1-002) = 3
    # Retrieved true pairs: S2-01 (1) + S2-02 (1) = 2
    # Pair recall: 2 / 3 = 0.6667
    assert metrics["number_of_s1_evaluated"] == 3
    assert metrics["total_true_pairs"] == 3
    assert metrics["retrieved_true_pairs"] == 2
    assert pytest.approx(metrics["pair_recall"], 0.001) == 2 / 3
    # Total candidates: 2 + 2 + 1 = 5 / 3 = 1.6667
    assert pytest.approx(metrics["average_candidates_per_s1"], 0.001) == 5 / 3


# ============================================================================
# 9. Float32 TF-IDF Matrix Verification (Item 2)
# ============================================================================

def test_tfidf_matrix_dtype_is_float32():
    """Verify that TF-IDF matrices use float32 dtype to halve memory usage."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-01"],
        "business_name": ["Acme Global Logistics"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-01"],
        "business_name": ["Acme Global Logistics Inc"],
    })

    captured_dtypes = []
    original_fit_transform = TfidfVectorizer.fit_transform

    def spy_fit_transform(self, *args, **kwargs):
        res = original_fit_transform(self, *args, **kwargs)
        captured_dtypes.append(res.dtype)
        return res

    with patch.object(TfidfVectorizer, "fit_transform", spy_fit_transform):
        candidates = retrieve_char_tfidf_candidates(s1_df, target_df, return_scores=True)

    assert len(captured_dtypes) > 0
    assert captured_dtypes[0] == np.float32

    # Also assert that returned similarity scores are valid floats
    assert "S2-01" in candidates["S1-01"]
    assert isinstance(candidates["S1-01"]["S2-01"], float)
    assert candidates["S1-01"]["S2-01"] > 0.0


# ============================================================================
# 10. min_df=2 Path on >100 Targets (Item 3)
# ============================================================================

def test_min_df_2_on_large_target_corpus():
    """Test the min_df=2 path on a target corpus exceeding 100 entities."""
    # Create 120 target records with repeated brand names so n-grams appear >= 2 times
    target_records = [
        {"entity_id": f"S2-{i:03d}", "business_name": f"Pioneer Tech Group Branch {i % 10}"}
        for i in range(120)
    ]
    target_df = pd.DataFrame(target_records)

    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Pioneer Tech Group"],
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5, min_df=2)

    assert "S1-001" in candidates
    assert len(candidates["S1-001"]) == 5
    # All retrieved candidates should belong to the Pioneer Tech Group branches
    for cid in candidates["S1-001"]:
        assert cid.startswith("S2-")


# ============================================================================
# 11. Synthetic Recall Comparison: min_df=1 vs min_df=2 (Item 3)
# ============================================================================

def test_synthetic_recall_comparison_min_df_1_vs_2():
    """Demonstrate why min_df=1 is strictly appropriate for ER candidate blocking recall.

    When a target corpus contains rare character patterns (e.g., unique foreign names,
    rare surnames, uncommon acronyms, or typos) appearing only once:
    - min_df=1 retains them in the vocabulary, yielding high similarity and 100% recall.
    - min_df=2 discards singleton character n-grams, resulting in 0 similarity and 0% recall.
    """
    # 120 generic target records
    target_records = [
        {"entity_id": f"S2-GEN-{i:03d}", "business_name": f"General Market Grocery {i}"}
        for i in range(120)
    ]
    # 1 rare target record with unique character sequences
    target_records.append({
        "entity_id": "S2-RARE",
        "business_name": "Xylophonic Quasar Tech LLC",
    })
    target_df = pd.DataFrame(target_records)

    s1_df = pd.DataFrame({
        "entity_id": ["S1-RARE"],
        "business_name": ["Xylophonic Quasar Tech"],
    })

    # Under min_df=1 (blocking default), the rare entity is successfully captured
    cands_df1 = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5, min_df=1)
    assert "S2-RARE" in cands_df1["S1-RARE"], "min_df=1 must capture rare character patterns"

    # Under min_df=2, singleton character n-grams are dropped, causing total recall failure
    cands_df2 = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5, min_df=2)
    assert "S2-RARE" not in cands_df2["S1-RARE"], "min_df=2 drops unique character patterns"


# ============================================================================
# 12. Robust NaN Handling in _extract_normalized_names (Item 4)
# ============================================================================

def test_extract_normalized_names_nan_handling():
    """Verify _extract_normalized_names safely handles None, NaN, pd.NA, and null strings."""
    dirty_values = [None, np.nan, pd.NA, float("nan"), "nan", "None", "null", "   ", "Valid Business Inc"]

    # Test 'name_normalized' column path
    df_norm = pd.DataFrame({"name_normalized": dirty_values})
    res_norm = _extract_normalized_names(df_norm)
    assert len(res_norm) == len(dirty_values)
    assert res_norm[:8] == ["", "", "", "", "", "", "", ""]
    assert res_norm[8] == "Valid Business Inc"

    # Test 'business_name' column path
    df_raw = pd.DataFrame({"business_name": dirty_values})
    res_raw = _extract_normalized_names(df_raw)
    assert len(res_raw) == len(dirty_values)
    assert res_raw[:8] == ["", "", "", "", "", "", "", ""]
    assert "valid business" in res_raw[8]


# ============================================================================
# 13. Duplicate Target Entity IDs Deterministically Keep Best Score (Item 5)
# ============================================================================

def test_duplicate_target_entity_ids_preserves_best_score():
    """Verify duplicate target entity IDs deterministically preserve the highest score."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Apex Global Logistics"],
    })

    # Target DF with duplicate entity_id 'S2-101':
    # - Row 0 has a weaker match ("Apex")
    # - Row 1 has an exact match ("Apex Global Logistics")
    target_df_order_a = pd.DataFrame({
        "entity_id": ["S2-101", "S2-101", "S2-202"],
        "business_name": [
            "Apex",
            "Apex Global Logistics",
            "Apex Other Logistics",
        ]
    })

    cands_a = retrieve_char_tfidf_candidates(s1_df, target_df_order_a, top_k=5, return_scores=True)
    score_a = cands_a["S1-001"]["S2-101"]

    # In reverse row order: exact match first, weaker match second
    target_df_order_b = pd.DataFrame({
        "entity_id": ["S2-101", "S2-101", "S2-202"],
        "business_name": [
            "Apex Global Logistics",
            "Apex",
            "Apex Other Logistics",
        ]
    })

    cands_b = retrieve_char_tfidf_candidates(s1_df, target_df_order_b, top_k=5, return_scores=True)
    score_b = cands_b["S1-001"]["S2-101"]

    # Both orderings must preserve the exact match score (near 1.0), never overwriting with weaker score
    assert pytest.approx(score_a, 0.001) == pytest.approx(score_b, 0.001)
    assert score_a > 0.95


# ============================================================================
# 14. return_scores=True Format & Range (Item 8)
# ============================================================================

def test_return_scores_format():
    """Verify return_scores=True returns a mapping of s1_id -> {candidate_id: float_score}."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Alpha Beta Gamma", "Delta Epsilon Zeta"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-001", "S3-002"],
        "business_name": ["Alpha Beta Gamma Corp", "Delta Epsilon Zeta LLC"],
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5, return_scores=True)

    assert isinstance(candidates, dict)
    assert set(candidates.keys()) == {"S1-001", "S1-002"}

    for s1_id, scores_map in candidates.items():
        assert isinstance(scores_map, dict)
        for cand_id, score in scores_map.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0001


# ============================================================================
# 15. min_similarity Filter (Item 8)
# ============================================================================

def test_min_similarity_filter():
    """Verify candidates below min_similarity are filtered out."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Quantum Robotics Technologies"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-HIGH", "S2-LOW"],
        "business_name": [
            "Quantum Robotics Technologies Inc",  # High similarity (> 0.8)
            "Quantum Bakery Cafe",                # Low similarity (< 0.5)
        ]
    })

    # With high min_similarity threshold (0.6)
    cands_filtered = retrieve_char_tfidf_candidates(
        s1_df, target_df, top_k=5, min_similarity=0.6, return_scores=True
    )

    assert "S2-HIGH" in cands_filtered["S1-001"]
    assert "S2-LOW" not in cands_filtered["S1-001"]


# ============================================================================
# 16. name_normalized Path (Item 8)
# ============================================================================

def test_name_normalized_path_used_directly():
    """Verify that when 'name_normalized' is present, it is used without re-normalization."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["RAW UNUSED NAME 1"],
        "name_normalized": ["custom normalized alpha"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-001"],
        "business_name": ["RAW UNUSED NAME 2"],
        "name_normalized": ["custom normalized alpha"],
    })

    candidates = retrieve_char_tfidf_candidates(s1_df, target_df, top_k=5)

    assert "S2-001" in candidates["S1-001"]


# ============================================================================
# 17. Empty Target DataFrame (Item 8)
# ============================================================================

def test_empty_target_dataframe():
    """Verify retrieval gracefully handles an empty target DataFrame."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Acme Corp", "Beta LLC"],
    })
    empty_target_df = pd.DataFrame(columns=["entity_id", "business_name"])

    # return_scores=False
    cands_sets = retrieve_char_tfidf_candidates(s1_df, empty_target_df, return_scores=False)
    assert cands_sets == {"S1-001": set(), "S1-002": set()}

    # return_scores=True
    cands_dicts = retrieve_char_tfidf_candidates(s1_df, empty_target_df, return_scores=True)
    assert cands_dicts == {"S1-001": {}, "S1-002": {}}


# ============================================================================
# 18. Missing entity_id Column Error (Item 8)
# ============================================================================

def test_missing_entity_id_raises_value_error():
    """Verify missing entity_id in s1_df or target_df raises ValueError."""
    bad_s1 = pd.DataFrame({"business_name": ["Acme"]})
    good_s1 = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Acme"]})
    bad_target = pd.DataFrame({"business_name": ["Acme Corp"]})
    good_target = pd.DataFrame({"entity_id": ["S2-1"], "business_name": ["Acme Corp"]})

    with pytest.raises(ValueError, match="s1_df must contain 'entity_id'"):
        retrieve_char_tfidf_candidates(bad_s1, good_target)

    with pytest.raises(ValueError, match="target_df must contain 'entity_id'"):
        retrieve_char_tfidf_candidates(good_s1, bad_target)


# ============================================================================
# 19. s1_ids_subset in Blocking Recall Evaluator (Item 8)
# ============================================================================

def test_evaluate_blocking_recall_s1_ids_subset(tmp_path: Path):
    """Verify evaluate_blocking_recall restricts evaluation to s1_ids_subset."""
    gt_file = tmp_path / "ground_truth.tsv"
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\tS2-002\n"
    )
    gt_file.write_text(gt_content, encoding="utf-8")

    candidates = {
        "S1-001": {"S2-001"},
        "S1-002": {"S2-999"},
    }

    # Evaluate ONLY S1-001
    metrics = evaluate_blocking_recall(candidates, gt_file, s1_ids_subset={"S1-001"})

    assert metrics["number_of_s1_evaluated"] == 1
    assert metrics["total_true_pairs"] == 1
    assert metrics["retrieved_true_pairs"] == 1
    assert metrics["pair_recall"] == 1.0


# ============================================================================
# 20. Evaluator ID Mismatch Error (Item 7 & 8)
# ============================================================================

def test_evaluate_blocking_recall_id_mismatch_raises_error(tmp_path: Path):
    """Verify evaluate_blocking_recall raises ValueError when candidate S1 IDs fail to join ground truth."""
    gt_file = tmp_path / "ground_truth.tsv"
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\tS2-002\n"
    )
    gt_file.write_text(gt_content, encoding="utf-8")

    # Candidates contain S1 IDs completely disjoint from ground truth
    mismatched_candidates = {
        "WRONG_ID_001": {"S2-001"},
        "WRONG_ID_002": {"S2-002"},
    }

    with pytest.raises(ValueError, match="Evaluation failed: None of the .* candidate S1 entity IDs joined"):
        evaluate_blocking_recall(mismatched_candidates, gt_file)


# ============================================================================
# 21. Explicit CSR Format Ensured (Item 6)
# ============================================================================

def test_explicit_csr_format_ensured():
    """Verify that matrices used for row-wise extraction are strictly scipy CSR format."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-01"],
        "business_name": ["Acme Corp"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-01"],
        "business_name": ["Acme Corporation"],
    })

    # Intercept dot product to verify row_sims is CSR
    original_dot = sp.csr_matrix.dot
    captured_sim_types = []

    def spy_dot(self, other):
        res = original_dot(self, other)
        captured_sim_types.append(type(res))
        return res

    with patch.object(sp.csr_matrix, "dot", spy_dot):
        retrieve_char_tfidf_candidates(s1_df, target_df)

    assert len(captured_sim_types) > 0
    for mat_type in captured_sim_types:
        assert issubclass(mat_type, sp.csr_matrix)


# ============================================================================
# 22. Word TF-IDF Placeholder (Architecture Guard)
# ============================================================================

def test_word_tfidf_raises_not_implemented():
    """Verify word TF-IDF retrieval placeholder raises NotImplementedError as scheduled."""
    with pytest.raises(NotImplementedError):
        retrieve_word_tfidf_candidates(pd.DataFrame(), pd.DataFrame())

