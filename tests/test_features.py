"""Unit tests for pairwise feature engineering module (src/features.py)."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    compute_string_similarities,
    extract_pair_features,
)


# ============================================================================
# 1. compute_string_similarities Tests
# ============================================================================

def test_compute_string_similarities_exact_and_partial():
    """Verify compute_string_similarities handles identical, partial, and empty strings."""
    # Identical
    res_exact = compute_string_similarities("Apex Global Logistics", "Apex Global Logistics")
    assert res_exact["exact_match"] == 1.0
    assert res_exact["char_ngram_sim"] == 1.0
    assert pytest.approx(res_exact["word_ngram_sim"], 0.001) == 1.0
    assert res_exact["token_overlap"] == 1.0
    assert res_exact["token_dice"] == 1.0
    assert res_exact["len_diff"] == 0.0
    assert res_exact["len_ratio"] == 1.0

    # Partial
    res_partial = compute_string_similarities("Apex Global Logistics Inc", "Apex Global Logistics")
    assert res_partial["exact_match"] == 0.0
    assert res_partial["char_ngram_sim"] > 0.8
    assert res_partial["word_ngram_sim"] > 0.7
    assert 0.0 < res_partial["token_overlap"] < 1.0
    assert res_partial["len_diff"] == 4.0

    # Empty
    res_empty = compute_string_similarities("", "Apex Global Logistics")
    assert res_empty["exact_match"] == 0.0
    assert res_empty["char_ngram_sim"] == 0.0
    assert res_empty["word_ngram_sim"] == 0.0
    assert res_empty["token_overlap"] == 0.0
    assert res_empty["len_diff"] == len("Apex Global Logistics")
    assert res_empty["len_ratio"] == 0.0


# ============================================================================
# 2. Exact Name and Address Match Test
# ============================================================================

def test_extract_pair_features_exact_name_address_match():
    """Verify exact match indicators and similarity scores when entities are identical."""
    s1_rec = {
        "business_name": "Apex Global Logistics Private Limited",
        "address": "100 Main Street, Suite 400, Chicago, IL 60601",
        "country": "US",
    }
    cand_rec = {
        "business_name": "Apex Global Logistics Private Limited",
        "address": "100 Main Street, Suite 400, Chicago, IL 60601",
        "country": "US",
    }

    feats = extract_pair_features(s1_rec, cand_rec)

    assert feats["name_exact_match"] == 1.0
    assert feats["name_compact_exact_match"] == 1.0
    assert feats["name_char_tfidf_sim"] == 1.0
    assert feats["name_word_tfidf_sim"] == 1.0
    assert feats["name_token_overlap"] == 1.0
    assert feats["name_token_dice"] == 1.0
    assert feats["name_len_diff"] == 0.0
    assert feats["name_len_ratio"] == 1.0
    assert feats["name_has_missing"] == 0.0

    assert feats["address_exact_match"] == 1.0
    assert feats["address_compact_exact_match"] == 1.0
    assert feats["address_string_sim"] == 1.0
    assert feats["address_token_overlap"] == 1.0
    assert feats["address_numeric_overlap"] == 1.0
    assert feats["address_numeric_exact_match"] == 1.0
    assert feats["address_numeric_mismatch"] == 0.0
    assert feats["address_len_diff"] == 0.0
    assert feats["address_has_missing"] == 0.0

    assert feats["country_exact_match"] == 1.0
    assert feats["country_mismatch"] == 0.0
    assert feats["country_has_missing"] == 0.0


# ============================================================================
# 3. Noisy Name and Address Test
# ============================================================================

def test_extract_pair_features_noisy_name_and_address():
    """Verify robust similarity scoring for noisy variations in business name and address."""
    s1_rec = {
        "business_name": "Apex Global Logistics Inc",
        "address": "18/A M.G. Road, Bangalore 560001",
        "country": "IN",
    }
    cand_rec = {
        "business_name": "Apex Global Logistics",
        "address": "18/A Mahatma Gandhi Rd, Bengaluru 560001",
        "country": "IN",
    }

    feats = extract_pair_features(s1_rec, cand_rec)

    # High similarity despite noise
    assert feats["name_char_tfidf_sim"] > 0.75
    assert feats["name_word_tfidf_sim"] > 0.7
    assert feats["name_token_overlap"] > 0.6

    # Address similarity and numeric token preservation
    assert feats["address_string_sim"] > 0.35
    assert feats["address_numeric_overlap"] == 1.0  # Both contain 18 and 560001
    assert feats["address_numeric_exact_match"] == 1.0
    assert feats["address_numeric_mismatch"] == 0.0

    assert feats["country_exact_match"] == 1.0
    assert feats["country_mismatch"] == 0.0


# ============================================================================
# 4. Missing Address Test
# ============================================================================

def test_extract_pair_features_missing_address():
    """Verify safe handling and indicator flags when address is None, NaN, or empty."""
    s1_rec = {
        "business_name": "Sterling Financial Advisors",
        "address": None,
        "country": "US",
    }
    cand_rec = {
        "business_name": "Sterling Financial Advisors",
        "address": "100 Wall Street, New York, NY 10005",
        "country": "US",
    }

    feats = extract_pair_features(s1_rec, cand_rec)

    assert feats["address_has_missing"] == 1.0
    assert feats["address_exact_match"] == 0.0
    assert feats["address_compact_exact_match"] == 0.0
    assert feats["address_string_sim"] == 0.0
    assert feats["address_token_overlap"] == 0.0
    assert feats["address_numeric_overlap"] == 0.0
    assert feats["address_numeric_exact_match"] == 0.0
    assert feats["address_numeric_mismatch"] == 0.0
    assert feats["address_len_diff"] > 0.0

    # Ensure all values are finite floats
    for k, v in feats.items():
        assert isinstance(v, (float, int))
        assert np.isfinite(v), f"Feature {k} is not finite: {v}"


# ============================================================================
# 5. Missing Name Test
# ============================================================================

def test_extract_pair_features_missing_name():
    """Verify safe handling when business name is None or empty."""
    s1_rec = {
        "business_name": None,
        "address": "100 Main St",
        "country": "US",
    }
    cand_rec = {
        "business_name": "Acme Corp",
        "address": "100 Main St",
        "country": "US",
    }

    feats = extract_pair_features(s1_rec, cand_rec)

    assert feats["name_has_missing"] == 1.0
    assert feats["name_exact_match"] == 0.0
    assert feats["name_compact_exact_match"] == 0.0
    assert feats["name_char_tfidf_sim"] == 0.0
    assert feats["name_word_tfidf_sim"] == 0.0
    assert feats["name_token_overlap"] == 0.0
    assert feats["name_len_ratio"] == 0.0

    for k, v in feats.items():
        assert np.isfinite(v), f"Feature {k} is not finite: {v}"


# ============================================================================
# 6. Country Match, Mismatch, and Missing Tests
# ============================================================================

def test_extract_pair_features_country_match_mismatch_missing():
    """Verify country matching, mismatch, and missingness flags."""
    base_s1 = {"business_name": "Test Co", "address": "Test Rd"}
    base_cand = {"business_name": "Test Co", "address": "Test Rd"}

    # Exact match
    f_match = extract_pair_features({**base_s1, "country": "IN"}, {**base_cand, "country": "in"})
    assert f_match["country_exact_match"] == 1.0
    assert f_match["country_mismatch"] == 0.0
    assert f_match["country_has_missing"] == 0.0

    # Mismatch
    f_mismatch = extract_pair_features({**base_s1, "country": "US"}, {**base_cand, "country": "IN"})
    assert f_mismatch["country_exact_match"] == 0.0
    assert f_mismatch["country_mismatch"] == 1.0
    assert f_mismatch["country_has_missing"] == 0.0

    # One missing
    f_one_missing = extract_pair_features({**base_s1, "country": "US"}, {**base_cand, "country": None})
    assert f_one_missing["country_exact_match"] == 0.0
    assert f_one_missing["country_mismatch"] == 0.0
    assert f_one_missing["country_has_missing"] == 1.0

    # Both missing
    f_both_missing = extract_pair_features({**base_s1, "country": ""}, {**base_cand, "country": np.nan})
    assert f_both_missing["country_exact_match"] == 0.0
    assert f_both_missing["country_mismatch"] == 0.0
    assert f_both_missing["country_has_missing"] == 1.0


# ============================================================================
# 7. Numeric Token Overlap Scenarios
# ============================================================================

def test_extract_pair_features_numeric_token_overlap():
    """Verify numeric token overlap across exact, partial, and disjoint cases."""
    # Exact numbers
    s1 = {"business_name": "A", "address": "Flat 402, Building 18, Street 5"}
    s2 = {"business_name": "A", "address": "Bldg 18, Flat 402, St 5"}
    f_exact = extract_pair_features(s1, s2)
    assert f_exact["address_numeric_overlap"] == 1.0
    assert f_exact["address_numeric_exact_match"] == 1.0
    assert f_exact["address_numeric_mismatch"] == 0.0

    # Partial number overlap
    s3 = {"business_name": "A", "address": "Flat 502, Building 18, Street 5"}
    f_part = extract_pair_features(s1, s3)
    assert 0.0 < f_part["address_numeric_overlap"] < 1.0
    assert f_part["address_numeric_exact_match"] == 0.0
    assert f_part["address_numeric_mismatch"] == 0.0

    # Completely disjoint numbers
    s4 = {"business_name": "A", "address": "Flat 999, Building 77, Street 88"}
    f_disjoint = extract_pair_features(s1, s4)
    assert f_disjoint["address_numeric_overlap"] == 0.0
    assert f_disjoint["address_numeric_exact_match"] == 0.0
    assert f_disjoint["address_numeric_mismatch"] == 1.0

    # No numbers in addresses
    s5 = {"business_name": "A", "address": "Main Street, Downtown Area"}
    s6 = {"business_name": "A", "address": "Main Street, Central Square"}
    f_no_nums = extract_pair_features(s5, s6)
    assert f_no_nums["address_numeric_overlap"] == 0.0
    assert f_no_nums["address_numeric_exact_match"] == 0.0
    assert f_no_nums["address_numeric_mismatch"] == 0.0


# ============================================================================
# 8. build_feature_matrix Output & No NaN / Inf
# ============================================================================

def test_build_feature_matrix_no_nan_or_inf():
    """Verify build_feature_matrix produces clean numerical outputs with zero NaN/inf values."""
    s1_lookup = {
        "S1-1": {"business_name": "Apex Global Logistics Inc", "address": "100 Main St", "country": "US"},
        "S1-2": {"business_name": None, "address": None, "country": None},
        "S1-3": {"business_name": "Sterling Financial", "address": "Wall St 42", "country": "US"},
    }
    target_lookup = {
        "S2-1": {"business_name": "Apex Global Logistics", "address": "100 Main St", "country": "US"},
        "S2-2": {"business_name": "Other Co", "address": "Other Rd", "country": "IN"},
        "S2-3": {"business_name": "Sterling Financial Advisors", "address": "Wall St 42", "country": "US"},
    }

    pairs = [("S1-1", "S2-1"), ("S1-2", "S2-2"), ("S1-3", "S2-3")]
    char_scores = {"S1-1": {"S2-1": 0.88}, "S1-3": {"S2-3": 0.95}}
    word_scores = {"S1-1": {"S2-1": 0.91}, "S1-3": {"S2-3": 0.89}}

    df = build_feature_matrix(
        candidate_pairs=pairs,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
        char_scores=char_scores,
        word_scores=word_scores,
    )

    assert len(df) == 3
    assert "source1_entity_id" in df.columns
    assert "matched_entity_id" in df.columns

    # Verify all required feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"

    # Verify strictly zero NaN and zero inf values
    assert df.isna().sum().sum() == 0, f"Found NaN values:\n{df.isna().sum()}"
    num_matrix = df[FEATURE_COLUMNS].to_numpy()
    assert np.isfinite(num_matrix).all(), "Found non-finite values (inf or -inf)"

    # Verify precomputed scores were applied
    assert pytest.approx(df.loc[0, "name_char_tfidf_sim"], 0.001) == 0.88
    assert pytest.approx(df.loc[0, "name_word_tfidf_sim"], 0.001) == 0.91


# ============================================================================
# 9. build_feature_matrix Input Formats
# ============================================================================

def test_build_feature_matrix_input_formats():
    """Verify build_feature_matrix supports DataFrames and dictionaries for lookups and pairs."""
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Alpha Corp", "Beta LLC"],
        "address": ["123 Alpha St", "456 Beta Rd"],
        "country": ["US", "IN"],
    })
    target_df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ["Alpha Corporation", "Beta Limited"],
        "address": ["123 Alpha Street", "456 Beta Road"],
        "country": ["US", "IN"],
    })

    pairs_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_id": ["S2-1", "S2-2"],
        "char_sim": [0.85, 0.75],
        "word_sim": [0.90, 0.80],
    })

    df = build_feature_matrix(
        candidate_pairs=pairs_df,
        s1_lookup=s1_df,
        target_lookup=target_df,
    )

    assert len(df) == 2
    assert list(df["source1_entity_id"]) == ["S1-1", "S1-2"]
    assert list(df["matched_entity_id"]) == ["S2-1", "S2-2"]
    assert pytest.approx(df.loc[0, "name_char_tfidf_sim"], 0.001) == 0.85
    assert pytest.approx(df.loc[0, "name_word_tfidf_sim"], 0.001) == 0.90


# ============================================================================
# 10. build_feature_matrix Chunked Processing
# ============================================================================

def test_build_feature_matrix_chunked_processing():
    """Verify chunked processing produces identical results to unchunked processing."""
    s1_lookup = {f"S1-{i}": {"business_name": f"Business {i}", "address": f"Address {i}"} for i in range(10)}
    target_lookup = {f"S2-{i}": {"business_name": f"Target {i}", "address": f"Address {i}"} for i in range(10)}
    pairs = [(f"S1-{i}", f"S2-{i}") for i in range(10)]

    df_full = build_feature_matrix(pairs, s1_lookup, target_lookup)
    df_chunked = build_feature_matrix(pairs, s1_lookup, target_lookup, chunk_size=3)

    assert len(df_full) == len(df_chunked) == 10
    pd.testing.assert_frame_equal(df_full, df_chunked)
