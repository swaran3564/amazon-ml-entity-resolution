"""Unit tests for src/model.py.

Covers:
- Model construction with baseline configuration and custom overrides
- Training on a synthetic dataset with class imbalance
- Probability prediction (range [0.0, 1.0], shape, strictly finite)
- Threshold prediction (binary 0/1 output, threshold sensitivity)
- F0.5 calculation (accuracy vs known values and sklearn fbeta_score)
- Threshold search over configurable grid
- Handling missing (NaN) and infinite (inf/-inf) feature values safely (no NaN/inf output)
- Automatic FEATURE_COLUMNS selection and explicit column lists
"""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.features import FEATURE_COLUMNS
from src.model import (
    EntityResolutionClassifier,
    ThresholdSearchResult,
    build_classifier,
    calculate_f05,
    compute_f_beta,
    optimize_f05_threshold,
    predict_matches,
    predict_proba,
    predict_with_threshold,
    sanitize_features,
    select_feature_columns,
    train_classifier,
)


@pytest.fixture
def synthetic_data():
    """Create a reproducible synthetic dataset for candidate matching tests."""
    rng = np.random.RandomState(42)
    n_samples = 40
    # 32 negatives, 8 positives (imbalance matching entity resolution candidate generation)
    y = np.array([0] * 32 + [1] * 8)

    # Positives have higher similarity scores
    f1_neg = rng.uniform(0.0, 0.4, size=32)
    f1_pos = rng.uniform(0.6, 1.0, size=8)
    f1 = np.concatenate([f1_neg, f1_pos])

    f2_neg = rng.uniform(0.0, 0.3, size=32)
    f2_pos = rng.uniform(0.5, 0.9, size=8)
    f2 = np.concatenate([f2_neg, f2_pos])

    X = pd.DataFrame({"feat_sim1": f1, "feat_sim2": f2})
    return X, y


# =========================================================================
# 1. Model Construction Tests
# =========================================================================


def test_build_classifier_defaults():
    """Verify build_classifier produces a valid LGBMClassifier with baseline parameters."""
    clf = build_classifier()
    assert isinstance(clf, LGBMClassifier)
    assert clf.random_state == 42
    assert clf.class_weight == "balanced"
    assert clf.objective == "binary"
    assert clf.learning_rate == 0.05
    assert clf.num_leaves == 31


def test_build_classifier_custom_parameters():
    """Verify custom parameters properly override defaults."""
    custom_params = {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 15,
    }
    clf = build_classifier(params=custom_params, class_weight=None, random_state=123)
    assert clf.n_estimators == 50
    assert clf.learning_rate == 0.1
    assert clf.num_leaves == 15
    assert clf.class_weight is None
    assert clf.random_state == 123


def test_entity_resolution_classifier_init():
    """Verify EntityResolutionClassifier initializes correctly."""
    model = EntityResolutionClassifier(
        params={"n_estimators": 30},
        threshold=0.6,
        class_weight="balanced",
        random_state=42,
    )
    assert model.threshold == 0.6
    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.model_ is None
    assert model.feature_columns_ is None


# =========================================================================
# 2. Training on Synthetic Dataset Tests
# =========================================================================


def test_train_classifier_synthetic_data(synthetic_data):
    """Verify train_classifier successfully trains on synthetic DataFrame and binary labels."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    assert isinstance(clf, EntityResolutionClassifier)
    assert clf.model_ is not None
    assert clf.feature_columns_ == ["feat_sim1", "feat_sim2"]

    # Ensure model is fitted
    assert hasattr(clf.model_, "n_features_in_")
    assert clf.model_.n_features_in_ == 2


@pytest.mark.filterwarnings("ignore:.*eval_set.*")
def test_fit_with_eval_set(synthetic_data):
    """Verify fitting works when an evaluation set is passed."""
    X, y = synthetic_data
    clf = EntityResolutionClassifier(params={"n_estimators": 10})
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.model_ is not None


# =========================================================================
# 3. Probability Prediction Tests
# =========================================================================


def test_predict_proba_range_and_shape(synthetic_data):
    """Verify predicted probabilities are 1D array bounded in [0.0, 1.0]."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    probs = clf.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.all(np.isfinite(probs))


def test_predict_proba_standalone_function(synthetic_data):
    """Verify the standalone predict_proba helper works identically."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    probs1 = clf.predict_proba(X)
    probs2 = predict_proba(clf, X)
    np.testing.assert_array_almost_equal(probs1, probs2)


def test_predict_proba_empty_dataframe():
    """Verify predict_proba safely returns an empty array for empty DataFrames."""
    clf = EntityResolutionClassifier()
    clf.model_ = build_classifier()
    clf.feature_columns_ = ["f1"]

    empty_df = pd.DataFrame(columns=["f1"])
    probs = clf.predict_proba(empty_df)
    assert len(probs) == 0
    assert isinstance(probs, np.ndarray)


# =========================================================================
# 4. Threshold Prediction Tests
# =========================================================================


def test_predict_matches_binary(synthetic_data):
    """Verify binary match predictions only contain 0 and 1."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    preds = clf.predict(X, threshold=0.5)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})

    # Standalone function
    func_preds = predict_matches(clf, X, threshold=0.5)
    np.testing.assert_array_equal(preds, func_preds)


def test_predict_threshold_monotonicity(synthetic_data):
    """Verify that a higher threshold produces fewer or equal positive predictions."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    preds_low = clf.predict(X, threshold=0.1)
    preds_high = clf.predict(X, threshold=0.9)

    assert np.sum(preds_low) >= np.sum(preds_high)


def test_predict_with_threshold_direct():
    """Verify predict_with_threshold logic on direct probability arrays."""
    probs = [0.1, 0.49, 0.50, 0.51, 0.95]
    preds = predict_with_threshold(probs, threshold=0.5)
    np.testing.assert_array_equal(preds, [0, 0, 1, 1, 1])


# =========================================================================
# 5. F0.5 Calculation Tests
# =========================================================================


def test_compute_f_beta_known_values():
    """Verify compute_f_beta calculates exact expected values."""
    # Perfect precision and recall
    assert compute_f_beta(1.0, 1.0, beta=0.5) == 1.0

    # Zero cases
    assert compute_f_beta(0.0, 0.0, beta=0.5) == 0.0
    assert compute_f_beta(0.0, 1.0, beta=0.5) == 0.0
    assert compute_f_beta(1.0, 0.0, beta=0.5) == 0.0

    # P=0.8, R=0.5 -> 1.25 * 0.4 / (0.25 * 0.8 + 0.5) = 0.5 / 0.7 = 5/7
    expected = 5.0 / 7.0
    assert np.isclose(compute_f_beta(0.8, 0.5, beta=0.5), expected)

    # Precision weighting check:
    # High precision (P=0.9, R=0.4) vs high recall (P=0.4, R=0.9)
    f05_high_p = compute_f_beta(0.9, 0.4, beta=0.5)
    f05_high_r = compute_f_beta(0.4, 0.9, beta=0.5)
    assert f05_high_p > f05_high_r, "F0.5 must favor precision over recall"


def test_calculate_f05_against_sklearn():
    """Verify calculate_f05 matches scikit-learn metrics exactly."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

    res = calculate_f05(y_true, y_pred, beta=0.5)

    expected_p = precision_score(y_true, y_pred, zero_division=0)
    expected_r = recall_score(y_true, y_pred, zero_division=0)
    expected_f = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

    assert np.isclose(res["precision"], expected_p)
    assert np.isclose(res["recall"], expected_r)
    assert np.isclose(res["f05"], expected_f)
    assert res["tp"] == 3
    assert res["fp"] == 1
    assert res["fn"] == 2
    assert res["tn"] == 4


# =========================================================================
# 6. Threshold Search Tests
# =========================================================================


def test_optimize_f05_threshold_grid():
    """Verify threshold search tests a grid and finds the optimal F0.5 threshold."""
    # Synthetic scenario: labels and probabilities
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.15, 0.2, 0.4, 0.45, 0.6, 0.55, 0.7, 0.85, 0.9])

    result = optimize_f05_threshold(y_true, y_probs)

    assert isinstance(result, ThresholdSearchResult)
    assert "best_threshold" in result
    assert "precision" in result
    assert "recall" in result
    assert "f05" in result
    assert "grid_results" in result

    # Best threshold should be around 0.65-0.7 to avoid false positives at 0.4-0.6
    assert 0.05 <= result.best_threshold <= 0.95
    assert result.f05 > 0.0
    assert result.precision > 0.0

    # Float casting
    assert float(result) == result.best_threshold
    assert result.best_threshold == result["best_threshold"]

    # Grid results structure
    assert len(result["grid_results"]) == 19
    for entry in result["grid_results"]:
        assert "threshold" in entry
        assert "f05" in entry
        assert "precision" in entry
        assert "recall" in entry


def test_optimize_f05_threshold_custom_grid():
    """Verify custom grid is respected."""
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.2, 0.3, 0.7, 0.8])

    custom_grid = [0.25, 0.5, 0.75]
    result = optimize_f05_threshold(y_true, y_probs, grid=custom_grid)

    assert result.best_threshold in custom_grid
    assert len(result["grid_results"]) == len(custom_grid)


def test_entity_resolution_classifier_optimize_threshold(synthetic_data):
    """Verify optimize_threshold updates classifier's threshold and applies to predictions."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    res = clf.optimize_threshold(X, y)
    assert clf.threshold == res.best_threshold
    assert clf.best_threshold_ == res.best_threshold

    # Prediction should use the calibrated threshold
    preds = clf.predict(X)
    eval_metrics = calculate_f05(y, preds)
    assert np.isclose(eval_metrics["f05"], res.f05)


# =========================================================================
# 7. No NaN/Inf Output and Safe Handling of Corrupt Values
# =========================================================================


def test_safe_handling_of_nan_and_inf_features():
    """Verify model safely trains and predicts with NaN, +inf, -inf in features."""
    rng = np.random.RandomState(42)
    n = 30
    y = np.array([0] * 24 + [1] * 6)

    f1 = rng.uniform(0.0, 1.0, size=n)
    f2 = rng.uniform(0.0, 1.0, size=n)

    # Introduce extreme values and NaNs
    f1[0] = np.nan
    f1[1] = np.inf
    f1[2] = -np.inf
    f2[3] = np.nan
    f2[4] = np.inf
    f2[5] = -np.inf

    X_dirty = pd.DataFrame({"f1": f1, "f2": f2})

    # Train model on dirty data
    clf = train_classifier(X_dirty, y)

    # Predict probabilities
    probs = clf.predict_proba(X_dirty)

    # Verify no NaN or inf in output probabilities
    assert not np.isnan(probs).any(), "Probabilities must not contain NaN"
    assert not np.isinf(probs).any(), "Probabilities must not contain Inf"
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Predict binary labels
    preds = clf.predict(X_dirty)
    assert not np.isnan(preds).any()
    assert set(np.unique(preds)).issubset({0, 1})


def test_sanitize_features_converts_inf():
    """Verify sanitize_features properly replaces inf with NaN and keeps finite numbers."""
    df = pd.DataFrame({
        "a": [1.0, np.inf, -np.inf, np.nan],
        "b": ["0.1", "0.2", "0.3", "0.4"],
    })
    clean_df, cols = sanitize_features(df)

    assert cols == ["a", "b"]
    assert not np.isinf(clean_df["a"]).any()
    assert clean_df["a"].isna().sum() == 3
    assert pd.api.types.is_numeric_dtype(clean_df["b"])


# =========================================================================
# 8. Feature Columns Selection Tests
# =========================================================================


def test_auto_select_feature_columns_with_standard_columns():
    """Verify that when FEATURE_COLUMNS are present, only they are selected."""
    # Create DataFrame with standard features plus metadata columns
    data = {col: [0.5, 0.8] for col in FEATURE_COLUMNS}
    data["source1_entity_id"] = ["s1_1", "s1_2"]
    data["matched_entity_id"] = ["s2_1", "s2_2"]
    data["extra_metadata"] = ["foo", "bar"]

    df = pd.DataFrame(data)
    selected = select_feature_columns(df)

    # Only FEATURE_COLUMNS should be selected, metadata excluded
    assert selected == FEATURE_COLUMNS
    assert "source1_entity_id" not in selected
    assert "matched_entity_id" not in selected
    assert "extra_metadata" not in selected


def test_explicit_feature_columns_override():
    """Verify explicit feature columns take priority."""
    data = {col: [0.5, 0.8] for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)

    explicit = ["name_exact_match", "address_exact_match"]
    selected = select_feature_columns(df, feature_columns=explicit)
    assert selected == explicit


def test_explicit_feature_columns_missing_raises():
    """Verify missing columns in explicit list raises ValueError."""
    df = pd.DataFrame({"col_a": [1, 2]})
    with pytest.raises(ValueError, match="Specified feature columns not found"):
        select_feature_columns(df, feature_columns=["col_nonexistent"])


def test_feature_importances(synthetic_data):
    """Verify get_feature_importances returns Series with feature names."""
    X, y = synthetic_data
    clf = train_classifier(X, y)

    importances = clf.get_feature_importances()
    assert isinstance(importances, pd.Series)
    assert set(importances.index) == {"feat_sim1", "feat_sim2"}
