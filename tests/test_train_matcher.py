"""Focused unit tests for scripts/train_matcher.py.

Covers:
- Multi-match labeling
- Singleton empty prediction scoring (F0.5 = 1.0)
- Singleton false-positive scoring (F0.5 = 0.0)
- Non-singleton empty prediction scoring (F0.5 = 0.0)
- Macro averaging across S1 entities
- Candidate ceiling recall (overall and by status)
- No pair leakage between train and validation partitions
- Status classification (singleton, matched_in_candidates, matched_but_missed)
"""

import numpy as np
import pandas as pd
import pytest

from scripts.train_matcher import (
    assign_candidate_labels,
    classify_s1_status,
    compute_entity_f05,
    compute_recall_ceilings,
    evaluate_entity_macro_f05,
    search_optimal_threshold,
    split_s1_entities,
)


# =========================================================================
# 1. Status Classification Tests
# =========================================================================


def test_status_classification():
    """Verify classification into singleton, matched_in_candidates, matched_but_missed."""
    # Singleton: empty ground truth
    assert classify_s1_status(set(), {"S2-1", "S2-2"}) == "singleton"
    assert classify_s1_status(set(), set()) == "singleton"

    # Matched in candidates: at least one GT match retrieved
    assert classify_s1_status({"S2-1"}, {"S2-1", "S2-2"}) == "matched_in_candidates"
    assert classify_s1_status({"S2-1", "S3-5"}, {"S2-1", "S2-99"}) == "matched_in_candidates"
    assert classify_s1_status({"S2-1", "S3-5"}, {"S3-5"}) == "matched_in_candidates"

    # Matched but missed: non-empty GT, but none retrieved
    assert classify_s1_status({"S2-1"}, {"S2-88", "S3-99"}) == "matched_but_missed"
    assert classify_s1_status({"S2-1", "S3-5"}, set()) == "matched_but_missed"


# =========================================================================
# 2. Multi-Match Labeling Tests
# =========================================================================


def test_multi_match_labeling():
    """Verify that every candidate appearing in GT set receives label=1 and others receive 0."""
    gt_map = {
        "S1-1": {"S2-10", "S2-20", "S3-30"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-40"},
    }

    pairs = [
        ("S1-1", "S2-10"),  # True match 1
        ("S1-1", "S2-20"),  # True match 2 (multi-match)
        ("S1-1", "S2-99"),  # Negative candidate
        ("S1-2", "S2-10"),  # Negative (singleton with candidates)
        ("S1-3", "S2-50"),  # Negative (missed true match)
    ]

    labels = assign_candidate_labels(pairs, gt_map)
    assert labels == [1, 1, 0, 0, 0]


# =========================================================================
# 3. Entity-Level F0.5 Scoring: Singletons and Empty Predictions
# =========================================================================


def test_singleton_empty_prediction_scoring():
    """Verify: truth empty + pred empty -> F0.5 = 1.0."""
    truth = set()
    pred = set()
    score = compute_entity_f05(truth, pred, beta=0.5)
    assert score == 1.0


def test_singleton_false_positive_scoring():
    """Verify: truth empty + pred non-empty -> F0.5 = 0.0."""
    truth = set()
    pred = {"S2-1"}
    score = compute_entity_f05(truth, pred, beta=0.5)
    assert score == 0.0

    pred_multi = {"S2-1", "S3-2"}
    assert compute_entity_f05(truth, pred_multi, beta=0.5) == 0.0


def test_non_singleton_empty_prediction_scoring():
    """Verify: truth non-empty + pred empty -> F0.5 = 0.0."""
    truth = {"S2-1"}
    pred = set()
    score = compute_entity_f05(truth, pred, beta=0.5)
    assert score == 0.0

    truth_multi = {"S2-1", "S2-2", "S3-3"}
    assert compute_entity_f05(truth_multi, pred, beta=0.5) == 0.0


def test_entity_f05_partial_and_perfect_matches():
    """Verify standard F0.5 scoring when both truth and pred are non-empty."""
    # Perfect match
    assert compute_entity_f05({"S2-1", "S2-2"}, {"S2-1", "S2-2"}, beta=0.5) == 1.0

    # No overlap -> 0.0
    assert compute_entity_f05({"S2-1"}, {"S2-2"}, beta=0.5) == 0.0

    # Truth: {A, B}, Pred: {A, C} -> TP=1, P=1/2, R=1/2 -> F0.5 = 0.5
    # Formula: 1.25 * 0.25 / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    assert np.isclose(compute_entity_f05({"A", "B"}, {"A", "C"}, beta=0.5), 0.5)

    # Precision weighted more than recall:
    # High precision: Truth {A, B}, Pred {A} -> P=1.0, R=0.5 -> 1.25 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333
    score_high_p = compute_entity_f05({"A", "B"}, {"A"}, beta=0.5)
    # High recall: Truth {A}, Pred {A, B} -> P=0.5, R=1.0 -> 1.25 * 0.5 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5556
    score_high_r = compute_entity_f05({"A"}, {"A", "B"}, beta=0.5)
    assert score_high_p > score_high_r, "F0.5 must reward high precision more than high recall"
    assert np.isclose(score_high_p, 5.0 / 6.0)
    assert np.isclose(score_high_r, 5.0 / 9.0)


# =========================================================================
# 4. Macro Averaging Across S1 Entities
# =========================================================================


def test_macro_averaging_across_s1():
    """Verify macro average computes the uniform mean of per-S1 scores."""
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    gt_map = {
        "S1-1": set(),            # singleton -> pred empty gives 1.0
        "S1-2": {"S2-A"},         # perfect match -> 1.0
        "S1-3": {"S2-B"},         # empty pred -> 0.0
    }
    pred_map = {
        "S1-1": set(),
        "S1-2": {"S2-A"},
        "S1-3": set(),
    }

    macro_score, per_s1 = evaluate_entity_macro_f05(s1_ids, gt_map, pred_map, beta=0.5)
    assert per_s1["S1-1"] == 1.0
    assert per_s1["S1-2"] == 1.0
    assert per_s1["S1-3"] == 0.0
    expected_macro = (1.0 + 1.0 + 0.0) / 3.0
    assert np.isclose(macro_score, expected_macro)


# =========================================================================
# 5. Candidate Ceiling Recall Tests
# =========================================================================


def test_candidate_ceiling_recall():
    """Verify candidate recall ceiling calculations overall and by status."""
    s1_ids = ["S1-1", "S1-2", "S1-3", "S1-4"]
    gt_map = {
        "S1-1": {"T1", "T2"},  # 2 true pairs, retrieves 1
        "S1-2": {"T3"},        # 1 true pair, retrieves 1
        "S1-3": {"T4", "T5"},  # 2 true pairs, retrieves 0 (matched_but_missed)
        "S1-4": set(),         # singleton
    }
    union_candidates = {
        "S1-1": {"T1", "X1"},
        "S1-2": {"T3", "X2"},
        "S1-3": {"X3", "X4"},
        "S1-4": {"X5"},
    }
    statuses = {
        "S1-1": "matched_in_candidates",
        "S1-2": "matched_in_candidates",
        "S1-3": "matched_but_missed",
        "S1-4": "singleton",
    }

    ceilings = compute_recall_ceilings(s1_ids, gt_map, union_candidates, statuses)

    # Total true pairs: 2 (S1-1) + 1 (S1-2) + 2 (S1-3) = 5
    # Retrieved true pairs: 1 (T1) + 1 (T3) = 2
    # Overall ceiling: 2 / 5 = 0.40
    assert ceilings["total_true_pairs"] == 5
    assert ceilings["retrieved_true_pairs"] == 2
    assert np.isclose(ceilings["overall_candidate_ceiling_recall"], 0.40)

    # Matched-in-candidates subset: S1-1 and S1-2 have 3 true pairs total, 2 retrieved -> 2/3 = 0.6667
    assert ceilings["matched_in_candidates_true_pairs"] == 3
    assert ceilings["matched_in_candidates_retrieved_pairs"] == 2
    assert np.isclose(ceilings["matched_in_candidates_ceiling_recall"], 2.0 / 3.0)

    # Matched-but-missed subset: 0 retrieved
    assert ceilings["matched_but_missed_ceiling_recall"] == 0.0


# =========================================================================
# 6. Zero Leakage Stratified Split Tests
# =========================================================================


def test_no_pair_leakage_between_train_and_val():
    """Verify that split_s1_entities strictly separates S1 entities with zero pair overlap."""
    rng = np.random.RandomState(42)
    s1_ids = [f"S1-{i:04d}" for i in range(100)]

    # 15 singletons, 65 matched_in_candidates, 20 matched_but_missed
    statuses = {}
    for i, sid in enumerate(s1_ids):
        if i < 15:
            statuses[sid] = "singleton"
        elif i < 80:
            statuses[sid] = "matched_in_candidates"
        else:
            statuses[sid] = "matched_but_missed"

    train_ids, val_ids = split_s1_entities(s1_ids, statuses, val_fraction=0.2, seed=42)

    assert len(train_ids) == 80
    assert len(val_ids) == 20

    # 1. Zero entity leakage
    assert set(train_ids).isdisjoint(set(val_ids)), "Train and Val entities must not overlap!"

    # 2. Simulate candidate pairs and verify zero pair leakage
    pairs = []
    for sid in s1_ids:
        for c in range(5):
            pairs.append((sid, f"Cand-{sid}-{c}"))
    pairs_df = pd.DataFrame(pairs, columns=["source1_entity_id", "matched_entity_id"])

    train_pairs = pairs_df[pairs_df["source1_entity_id"].isin(set(train_ids))]
    val_pairs = pairs_df[pairs_df["source1_entity_id"].isin(set(val_ids))]

    assert set(train_pairs["source1_entity_id"]).isdisjoint(set(val_pairs["source1_entity_id"]))
    assert len(train_pairs) + len(val_pairs) == len(pairs_df)

    # 3. Stratification preserved in val
    val_statuses = [statuses[sid] for sid in val_ids]
    val_status_counts = pd.Series(val_statuses).value_counts().to_dict()
    assert val_status_counts["singleton"] == 3  # 15 * 0.2
    assert val_status_counts["matched_in_candidates"] == 13  # 65 * 0.2
    assert val_status_counts["matched_but_missed"] == 4  # 20 * 0.2


# =========================================================================
# 7. Threshold Search Tests
# =========================================================================


def test_search_optimal_threshold():
    """Verify threshold search chooses threshold maximizing macro F0.5."""
    val_s1_ids = ["S1-1", "S1-2", "S1-3"]
    gt_map = {
        "S1-1": {"S2-A"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-B"},
    }

    # Pairs with probabilities
    # At t=0.6:
    # S1-1: S2-A (0.85) -> pred {S2-A} -> F0.5 = 1.0
    # S1-2: S2-C (0.45) -> pred {} -> F0.5 = 1.0
    # S1-3: S2-B (0.75) -> pred {S2-B} -> F0.5 = 1.0
    # Macro F0.5 = 1.0 at t=0.6!
    # But at t=0.3:
    # S1-2 gets false positive S2-C (0.45 >= 0.3) -> F0.5 = 0.0
    val_pairs_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_id": ["S2-A", "S2-C", "S2-B"],
    })
    val_probs = np.array([0.85, 0.45, 0.75])

    thresholds = [0.3, 0.6, 0.9]
    best_t, best_metrics, grid = search_optimal_threshold(
        val_s1_ids=val_s1_ids,
        val_pairs_df=val_pairs_df,
        val_probs=val_probs,
        gt_map=gt_map,
        thresholds=thresholds,
    )

    assert best_t == 0.6
    assert best_metrics["macro_f05"] == 1.0
    assert len(grid) == 3
