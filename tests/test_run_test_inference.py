"""Unit tests for the production test inference pipeline."""

import tempfile
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from scripts.run_test_inference import (
    load_model_and_vectorizers,
    merge_worker_outputs,
    normalize_country,
    retrieve_union_candidates_for_batch,
    run_test_inference,
)
from src.features import FEATURE_COLUMNS
from utils.validate_submission import validate


@pytest.fixture
def synthetic_test_env():
    """Create a minimal synthetic test environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)

        # 1. Create Source 1 test data (3 entities)
        s1_df = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002", "S1-003"],
            "business_name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
            "business_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
            "country": ["US", "US", "France"],
        })
        s1_path = test_dir / "test_source1.tsv"
        s1_df.to_csv(s1_path, sep="\t", index=False)

        # 2. Create Source 2 test data (3 entities)
        s2_df = pd.DataFrame({
            "entity_id": ["S2-001", "S2-002", "S2-003"],
            "business_name": ["Acme Corporation", "Beta Limited", "Zeta Enterprise"],
            "business_address": ["123 Main Street", "456 Oak Avenue", "100 Broadway"],
            "country": ["US", "US", "US"],
        })
        s2_path = test_dir / "test_source2.tsv"
        s2_df.to_csv(s2_path, sep="\t", index=False)

        # 3. Create Source 3 test data (2 entities)
        s3_df = pd.DataFrame({
            "entity_id": ["S3-001", "S3-002"],
            "business_name": ["Acme Holdings", "Omega Partners"],
            "business_address": ["123 Main St Ste 4", "999 High St"],
            "country": ["US", "France"],
        })
        s3_path = test_dir / "test_source3.tsv"
        s3_df.to_csv(s3_path, sep="\t", index=False)

        # 4. Create and save synthetic vectorizers
        all_names = list(s2_df["business_name"]) + list(s3_df["business_name"])
        char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
        char_vec.fit(all_names)
        joblib.dump(char_vec, cache_dir / "char_vectorizer.joblib")

        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), dtype=np.float32)
        word_vec.fit(all_names)
        joblib.dump(word_vec, cache_dir / "word_vectorizer.joblib")

        # 5. Create a minimal LightGBM model
        X_dummy = np.random.RandomState(42).randn(10, len(FEATURE_COLUMNS))
        y_dummy = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        lgb_data = lgb.Dataset(X_dummy, label=y_dummy, feature_name=FEATURE_COLUMNS)
        booster = lgb.train(
            {"objective": "binary", "verbose": -1, "num_leaves": 5},
            lgb_data,
            num_boost_round=5,
        )
        model_path = out_dir / "matcher_model_k40.txt"
        booster.save_model(str(model_path))

        yield {
            "test_dir": test_dir,
            "s1_path": s1_path,
            "s2_path": s2_path,
            "s3_path": s3_path,
            "model_path": model_path,
            "cache_dir": cache_dir,
            "out_dir": out_dir,
        }


def test_country_normalization():
    """Test 1: Comprehensive country normalization behavior."""
    assert normalize_country("US") == "us"
    assert normalize_country("  India  ") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("FRANCE") == "france"
    assert normalize_country("united states") == "united states"
    assert normalize_country(None) == ""
    assert normalize_country(np.nan) == ""
    assert normalize_country("") == ""
    assert normalize_country("   ") == ""
    assert normalize_country("nan") == ""
    assert normalize_country("NaN") == ""
    assert normalize_country("None") == ""
    assert normalize_country("null") == ""


def test_country_shard_isolation(synthetic_test_env):
    """Test 2: Queries are isolated strictly to their matching country partition."""
    env = synthetic_test_env
    cand_out = env["out_dir"] / "cand_iso.tsv"
    match_out = env["out_dir"] / "match_iso.tsv"

    run_test_inference(
        s1_path=env["s1_path"],
        s2_path=env["s2_path"],
        s3_path=env["s3_path"],
        model_path=env["model_path"],
        cache_dir=env["cache_dir"],
        candidate_output_path=cand_out,
        matching_output_path=match_out,
        threshold=0.1,
        char_top_k=5,
        word_top_k=5,
        target_chunk_size=10,
        s1_chunk_size=5,
        num_workers=1,
    )

    cand_df = pd.read_csv(cand_out, sep="\t", dtype=str)
    s2_df = pd.read_csv(env["s2_path"], sep="\t", dtype=str)
    s3_df = pd.read_csv(env["s3_path"], sep="\t", dtype=str)
    target_countries = pd.concat([s2_df, s3_df]).set_index("entity_id")["country"].to_dict()

    # S1-003 is France, so all its candidates must be France
    s1_003_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-003", "candidate_entity_ids"].values[0]
    if pd.notna(s1_003_cands) and s1_003_cands.strip():
        for cid in s1_003_cands.split(","):
            assert target_countries[cid] == "France", f"Expected France for S1-003 candidate {cid}, got {target_countries[cid]}"

    # S1-001 and S1-002 are US, so all their candidates must be US
    for sid in ["S1-001", "S1-002"]:
        s_cands = cand_df.loc[cand_df["source1_entity_id"] == sid, "candidate_entity_ids"].values[0]
        if pd.notna(s_cands) and s_cands.strip():
            for cid in s_cands.split(","):
                assert target_countries[cid] == "US", f"Expected US for {sid} candidate {cid}, got {target_countries[cid]}"


def test_missing_country_fallback(synthetic_test_env):
    """Test 3: S1 query with missing/empty country falls back safely to match targets."""
    env = synthetic_test_env
    # Create S1 with an empty country
    s1_fb_df = pd.DataFrame({
        "entity_id": ["S1-999"],
        "business_name": ["Acme Corp"],
        "business_address": ["123 Main St"],
        "country": [""],  # Empty country
    })
    s1_fb_path = env["test_dir"] / "test_s1_fallback.tsv"
    s1_fb_df.to_csv(s1_fb_path, sep="\t", index=False)

    cand_out = env["out_dir"] / "cand_fb.tsv"
    match_out = env["out_dir"] / "match_fb.tsv"

    run_test_inference(
        s1_path=s1_fb_path,
        s2_path=env["s2_path"],
        s3_path=env["s3_path"],
        model_path=env["model_path"],
        cache_dir=env["cache_dir"],
        candidate_output_path=cand_out,
        matching_output_path=match_out,
        threshold=0.5,
        char_top_k=5,
        word_top_k=5,
        target_chunk_size=10,
        s1_chunk_size=5,
    )

    cand_df = pd.read_csv(cand_out, sep="\t", dtype=str)
    assert len(cand_df) == 1
    assert cand_df.iloc[0]["source1_entity_id"] == "S1-999"
    # Should have retrieved Acme candidates via fallback
    cands = cand_df.iloc[0]["candidate_entity_ids"]
    assert pd.notna(cands) and len(cands) > 0


def test_cross_worker_duplicate_reconciliation():
    """Test 4: Merging deduplicates candidates/matches and preserves order."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        shard1_c = tmp_p / "w1_c.tsv"
        shard1_m = tmp_p / "w1_m.tsv"
        shard2_c = tmp_p / "w2_c.tsv"
        shard2_m = tmp_p / "w2_m.tsv"

        # Worker 1 returns S2-001, S2-002
        shard1_c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-001,S2-002\n")
        shard1_m.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-001\n")

        # Worker 2 returns duplicate S2-002, S2-003
        shard2_c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-002,S2-003\n")
        shard2_m.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-001,S2-002\n")

        final_c = tmp_p / "final_c.tsv"
        final_m = tmp_p / "final_m.tsv"

        stats = merge_worker_outputs(
            shard_cand_paths=[shard1_c, shard2_c],
            shard_match_paths=[shard1_m, shard2_m],
            required_s1_ids=["S1-01"],
            final_cand_path=final_c,
            final_match_path=final_m,
        )

        c_df = pd.read_csv(final_c, sep="\t", dtype=str)
        m_df = pd.read_csv(final_m, sep="\t", dtype=str)

        # Candidates should be deduplicated: S2-001, S2-002, S2-003
        assert c_df.iloc[0]["candidate_entity_ids"] == "S2-001,S2-002,S2-003"
        # Matches should be deduplicated: S2-001, S2-002
        assert m_df.iloc[0]["matched_entity_ids"] == "S2-001,S2-002"
        assert stats["subset_violations"] == 0


def test_exactly_one_row_per_s1_and_empty_handling():
    """Test 5 & 6: Exactly one row per S1 and empty candidate/match list handling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        shard_c = tmp_p / "w_c.tsv"
        shard_m = tmp_p / "w_m.tsv"

        # S1-A has candidates, S1-B has no candidates
        shard_c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-A\tS2-100\nS1-B\t\n")
        shard_m.write_text("source1_entity_id\tmatched_entity_ids\nS1-A\t\nS1-B\t\n")

        final_c = tmp_p / "final_c.tsv"
        final_m = tmp_p / "final_m.tsv"

        stats = merge_worker_outputs(
            shard_cand_paths=[shard_c],
            shard_match_paths=[shard_m],
            required_s1_ids=["S1-A", "S1-B", "S1-C"],  # S1-C wasn't even in shard
            final_cand_path=final_c,
            final_match_path=final_m,
        )

        assert stats["total_s1_rows"] == 3
        assert stats["zero_candidate_s1_count"] == 2  # S1-B and S1-C
        assert stats["zero_match_s1_count"] == 3      # all zero matches

        c_df = pd.read_csv(final_c, sep="\t", keep_default_na=False)
        m_df = pd.read_csv(final_m, sep="\t", keep_default_na=False)

        assert len(c_df) == 3
        assert len(m_df) == 3
        assert list(c_df["source1_entity_id"]) == ["S1-A", "S1-B", "S1-C"]
        assert c_df.loc[c_df["source1_entity_id"] == "S1-B", "candidate_entity_ids"].values[0] == ""
        assert c_df.loc[c_df["source1_entity_id"] == "S1-C", "candidate_entity_ids"].values[0] == ""


def test_candidate_subset_invariant():
    """Test 7: Matched IDs not present in candidates are filtered out by merger."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        shard_c = tmp_p / "w_c.tsv"
        shard_m = tmp_p / "w_m.tsv"

        # S1-X has candidates [S2-01], but rogue match contains [S2-01, S2-99]
        shard_c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-X\tS2-01\n")
        shard_m.write_text("source1_entity_id\tmatched_entity_ids\nS1-X\tS2-01,S2-99\n")

        final_c = tmp_p / "final_c.tsv"
        final_m = tmp_p / "final_m.tsv"

        merge_worker_outputs(
            shard_cand_paths=[shard_c],
            shard_match_paths=[shard_m],
            required_s1_ids=["S1-X"],
            final_cand_path=final_c,
            final_match_path=final_m,
        )

        m_df = pd.read_csv(final_m, sep="\t", keep_default_na=False)
        # S2-99 should NOT appear in matches because it is not in candidates
        assert m_df.iloc[0]["matched_entity_ids"] == "S2-01"


def test_deterministic_merge_ordering():
    """Test 8: Output order matches required_s1_ids regardless of shard input order."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        shard1 = tmp_p / "s1.tsv"
        shard2 = tmp_p / "s2.tsv"

        shard1.write_text("source1_entity_id\tcandidate_entity_ids\nS1-03\tS2-03\nS1-01\tS2-01\n")
        shard2.write_text("source1_entity_id\tcandidate_entity_ids\nS1-02\tS2-02\n")

        final_c = tmp_p / "final_c.tsv"
        final_m = tmp_p / "final_m.tsv"

        merge_worker_outputs(
            shard_cand_paths=[shard1, shard2],
            shard_match_paths=[shard1, shard2],
            required_s1_ids=["S1-01", "S1-02", "S1-03"],
            final_cand_path=final_c,
            final_match_path=final_m,
        )

        c_df = pd.read_csv(final_c, sep="\t")
        assert list(c_df["source1_entity_id"]) == ["S1-01", "S1-02", "S1-03"]


def test_end_to_end_test_inference(synthetic_test_env):
    """Test 9: Full execution of run_test_inference and official validator check."""
    env = synthetic_test_env
    cand_out = env["out_dir"] / "candidate_pairs.tsv"
    match_out = env["out_dir"] / "matching_results.tsv"

    results = run_test_inference(
        s1_path=env["s1_path"],
        s2_path=env["s2_path"],
        s3_path=env["s3_path"],
        model_path=env["model_path"],
        cache_dir=env["cache_dir"],
        candidate_output_path=cand_out,
        matching_output_path=match_out,
        threshold=0.5,
        char_top_k=5,
        word_top_k=5,
        target_chunk_size=10,
        s1_chunk_size=2,
        num_workers=1,
    )

    assert cand_out.exists()
    assert match_out.exists()
    assert results["number_of_test_s1_entities"] == 3

    cand_df = pd.read_csv(cand_out, sep="\t", dtype=str)
    assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert len(cand_df) == 3
    assert set(cand_df["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003"}

    match_df = pd.read_csv(match_out, sep="\t", dtype=str)
    assert list(match_df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(match_df) == 3
    assert set(match_df["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003"}

    # Official validator
    errors, warnings = validate(
        matching_path=str(match_out),
        candidate_path=str(cand_out),
        test_dir=str(env["test_dir"]),
        check_ids=False,
    )
    assert len(errors) == 0, f"Validator found errors: {errors}"
