"""Model training, threshold optimization, and inference module for Business Entity Resolution.

In the locked architecture:
- Exactly ONE downstream LightGBM classifier scoring candidate pairs.
- Candidate pairs receive ALL pairwise similarity features.
- Model trained with class imbalance handling (class_weight='balanced').
- Predictions scored against an F0.5 optimal decision threshold (precision emphasized over recall).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

try:
    from src.config import DEFAULT_THRESHOLD, LGBM_PARAMS
except ImportError:
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    DEFAULT_THRESHOLD = 0.5

try:
    from src.features import FEATURE_COLUMNS
except ImportError:
    FEATURE_COLUMNS = []


class ThresholdSearchResult(dict):
    """Container for threshold search results, subclassing dict for seamless access.

    Attributes / Keys
    -----------------
    best_threshold : float
        Probability threshold that maximizes the F_beta score.
    precision : float
        Precision achieved at the best threshold.
    recall : float
        Recall achieved at the best threshold.
    f05 : float
        F0.5 score achieved at the best threshold.
    f_beta : float
        Alias for f05 when beta=0.5.
    grid_results : list of dict
        Full evaluation results across all tested grid thresholds.
    """

    @property
    def best_threshold(self) -> float:
        return self["best_threshold"]

    @property
    def precision(self) -> float:
        return self["precision"]

    @property
    def recall(self) -> float:
        return self["recall"]

    @property
    def f05(self) -> float:
        return self["f05"]

    @property
    def f_beta(self) -> float:
        return self.get("f_beta", self["f05"])

    def __float__(self) -> float:
        return float(self["best_threshold"])

    def __repr__(self) -> str:
        return (
            f"ThresholdSearchResult(best_threshold={self['best_threshold']:.4f}, "
            f"precision={self['precision']:.4f}, recall={self['recall']:.4f}, "
            f"f05={self['f05']:.4f})"
        )


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score from precision and recall (defaults to F_0.5).

    Formula:
        (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    When beta=0.5:
        1.25 * (precision * recall) / (0.25 * precision + recall)
        Precision is weighted 4x as heavily as recall.

    Parameters
    ----------
    precision : float
        Precision score in [0.0, 1.0].
    recall : float
        Recall score in [0.0, 1.0].
    beta : float, default=0.5
        Weight parameter for precision vs recall.

    Returns
    -------
    float
        Computed F_beta score in [0.0, 1.0].
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)


def calculate_f05(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_pred: Union[np.ndarray, pd.Series, Sequence],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Calculate precision, recall, and F0.5 score for binary predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    beta : float, default=0.5
        Beta parameter for F-beta calculation.

    Returns
    -------
    dict
        Dictionary containing 'precision', 'recall', 'f05', 'f_beta', 'tp', 'fp', 'fn', 'tn'.
    """
    y_true_arr = (np.asarray(y_true).ravel() > 0).astype(int)
    y_pred_arr = (np.asarray(y_pred).ravel() > 0).astype(int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, y_pred has {len(y_pred_arr)} items."
        )

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f_score = compute_f_beta(precision, recall, beta=beta)

    return {
        "precision": precision,
        "recall": recall,
        "f05": f_score,
        "f_beta": f_score,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def select_feature_columns(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Determine the list of feature columns to use from a DataFrame.

    If `feature_columns` is explicitly provided, verifies and returns those columns.
    If `feature_columns` is None:
      - Automatically detects and returns existing columns matching `src.features.FEATURE_COLUMNS`.
      - If none of FEATURE_COLUMNS are present, falls back to all numeric columns in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : sequence of str, optional
        Explicit list of feature columns.

    Returns
    -------
    list of str
        Selected feature column names.
    """
    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Specified feature columns not found in DataFrame: {missing}")
        return list(feature_columns)

    if FEATURE_COLUMNS:
        present = [col for col in FEATURE_COLUMNS if col in df.columns]
        if present:
            return present

    return list(df.columns)


def sanitize_features(
    X: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean feature DataFrame for safe ingestion by LightGBM.

    - Selects feature columns (either explicit, or matching FEATURE_COLUMNS, or all columns).
    - Converts columns to numeric values safely.
    - Replaces infinite values (+inf, -inf) with NaN, which LightGBM handles natively.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    feature_columns : sequence of str, optional
        Explicit column names to select.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        The sanitized DataFrame and the list of selected feature column names.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    cols = select_feature_columns(X, feature_columns)
    if not cols:
        raise ValueError("No feature columns could be determined from the provided DataFrame.")

    X_clean = X[cols].copy()

    for col in cols:
        if not pd.api.types.is_numeric_dtype(X_clean[col]):
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")

    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X_clean, cols


def build_classifier(
    params: Optional[Dict[str, Any]] = None,
    class_weight: Optional[Union[str, Dict]] = "balanced",
    random_state: int = 42,
    **kwargs,
) -> LGBMClassifier:
    """Construct an LGBMClassifier with reproducible baseline settings.

    Parameters
    ----------
    params : dict, optional
        Custom LightGBM parameters overriding defaults.
    class_weight : str or dict, optional, default='balanced'
        Class weighting strategy to handle candidate pair imbalance (negatives >> positives).
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs :
        Additional keyword arguments passed to LGBMClassifier.

    Returns
    -------
    LGBMClassifier
        Configured, un-fitted LightGBM classifier.
    """
    base_params: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": class_weight,
        "random_state": random_state,
        "n_estimators": 100,
        "n_jobs": -1,
        "verbose": -1,
    }

    if LGBM_PARAMS:
        base_params.update(LGBM_PARAMS)

    # Explicit arguments override base config
    base_params["class_weight"] = class_weight
    base_params["random_state"] = random_state

    if params:
        base_params.update(params)
    if kwargs:
        base_params.update(kwargs)

    return LGBMClassifier(**base_params)


def predict_proba(
    model: Union[LGBMClassifier, Any],
    X: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Predict probabilities of positive class (P(y=1|X)).

    Guarantees strictly finite probabilities bounded in [0.0, 1.0] with no NaN or inf.

    Parameters
    ----------
    model : LGBMClassifier or EntityResolutionClassifier
        Trained binary classifier.
    X : pd.DataFrame
        Input features.
    feature_columns : sequence of str, optional
        Explicit feature columns to extract from X.

    Returns
    -------
    np.ndarray
        1D float array of probabilities in range [0.0, 1.0].
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)

    if hasattr(model, "model_") and hasattr(model, "feature_columns_"):
        return model.predict_proba(X)

    if feature_columns is None and hasattr(model, "feature_name_") and model.feature_name_:
        feature_columns = model.feature_name_

    X_clean, _ = sanitize_features(X, feature_columns=feature_columns)

    raw_probs = model.predict_proba(X_clean)
    if raw_probs.ndim == 2 and raw_probs.shape[1] >= 2:
        probs = raw_probs[:, 1]
    elif raw_probs.ndim == 2 and raw_probs.shape[1] == 1:
        probs = raw_probs[:, 0]
    else:
        probs = raw_probs.ravel()

    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    return probs.astype(np.float32)


def predict_with_threshold(
    probabilities: Union[np.ndarray, Sequence[float]],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Generate binary 0/1 predictions by applying a decision threshold to probabilities.

    Parameters
    ----------
    probabilities : array-like
        Predicted probabilities.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    np.ndarray
        1D integer array of binary predictions (0 or 1).
    """
    probs = np.asarray(probabilities)
    if len(probs) == 0:
        return np.array([], dtype=int)
    probs_clean = np.nan_to_num(probs, nan=0.0)
    return (probs_clean >= threshold).astype(int)


def predict_matches(
    model: Union[LGBMClassifier, Any],
    X_test: pd.DataFrame,
    threshold: Optional[float] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Generate binary match predictions using calibrated threshold.

    Parameters
    ----------
    model : LGBMClassifier or EntityResolutionClassifier
        Trained model.
    X_test : pd.DataFrame
        Candidate pair feature DataFrame.
    threshold : float, optional
        Classification threshold. If None, uses model.threshold or DEFAULT_THRESHOLD (0.5).
    feature_columns : sequence of str, optional
        Explicit feature columns to use.

    Returns
    -------
    np.ndarray
        1D integer array of binary predictions (0 or 1).
    """
    if len(X_test) == 0:
        return np.array([], dtype=int)

    if threshold is None:
        threshold = getattr(model, "threshold", DEFAULT_THRESHOLD)

    probs = predict_proba(model, X_test, feature_columns=feature_columns)
    return predict_with_threshold(probs, threshold=threshold)


def optimize_f05_threshold(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_probs: Union[np.ndarray, pd.Series, Sequence],
    grid: Optional[Sequence[float]] = None,
    beta: float = 0.5,
) -> ThresholdSearchResult:
    """Find the optimal probability threshold maximizing F_beta score (defaults to F0.5).

    Tests a grid of decision thresholds and returns the one achieving the highest
    F0.5 score. Precision is prioritized over recall.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0 or 1).
    y_probs : array-like
        Predicted probabilities for the positive class.
    grid : sequence of float, optional
        Candidate threshold values. Defaults to 19 values from 0.05 to 0.95 (step 0.05).
    beta : float, default=0.5
        F-score beta parameter (0.5 emphasizes precision).

    Returns
    -------
    ThresholdSearchResult (dict)
        Contains:
        - "best_threshold": float
        - "precision": float
        - "recall": float
        - "f05": float
        - "f_beta": float
        - "grid_results": list of per-threshold metrics
    """
    if grid is None:
        grid = [round(float(t), 4) for t in np.linspace(0.05, 0.95, 19)]
    else:
        grid = [float(t) for t in grid]

    y_true_arr = (np.asarray(y_true).ravel() > 0).astype(int)
    y_probs_arr = np.nan_to_num(np.asarray(y_probs).ravel(), nan=0.0)

    if len(y_true_arr) != len(y_probs_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, y_probs has {len(y_probs_arr)} items."
        )

    grid_results = []
    best_threshold = 0.5
    best_f05 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for t in grid:
        y_pred = (y_probs_arr >= t).astype(int)
        metrics = calculate_f05(y_true_arr, y_pred, beta=beta)
        f_score = metrics["f05"]
        p = metrics["precision"]
        r = metrics["recall"]

        grid_results.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f05": float(f_score),
            "f_beta": float(f_score),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tn": int(metrics["tn"]),
        })

        is_better = False
        if f_score > best_f05:
            is_better = True
        elif np.isclose(f_score, best_f05, atol=1e-7):
            if p > best_precision:
                is_better = True
            elif np.isclose(p, best_precision, atol=1e-7) and t > best_threshold:
                is_better = True

        if is_better:
            best_f05 = f_score
            best_threshold = float(t)
            best_precision = float(p)
            best_recall = float(r)

    if best_f05 < 0.0:
        best_threshold = 0.5
        best_f05 = 0.0
        best_precision = 0.0
        best_recall = 0.0

    return ThresholdSearchResult({
        "best_threshold": float(best_threshold),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f05": float(best_f05),
        "f_beta": float(best_f05),
        "grid_results": grid_results,
    })


class EntityResolutionClassifier:
    """Wrapper around LightGBM classifier specifically tuned for Entity Resolution.

    Features:
    - Automatically handles feature column selection from DataFrame.
    - Sanitizes missing (NaN) and infinite (inf/-inf) values safely.
    - Supports class weighting (default 'balanced') for extreme candidate imbalance.
    - Calibrates and maintains decision threshold for F0.5 metric.
    - Guaranteed strictly finite outputs (no NaN or inf).
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        class_weight: Optional[Union[str, Dict]] = "balanced",
        random_state: int = 42,
        **kwargs,
    ):
        self.params = params or {}
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.threshold = float(threshold)
        self.class_weight = class_weight
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_: Optional[LGBMClassifier] = None
        self.feature_columns_: Optional[List[str]] = None
        self.best_threshold_: Optional[float] = None
        self.threshold_search_result_: Optional[ThresholdSearchResult] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, Sequence],
        feature_columns: Optional[Sequence[str]] = None,
        eval_set: Optional[List[Tuple[pd.DataFrame, Any]]] = None,
        **fit_params,
    ) -> "EntityResolutionClassifier":
        """Fit the LightGBM classifier on training features and binary target."""
        target_cols = feature_columns or self.feature_columns
        X_clean, cols_used = sanitize_features(X, feature_columns=target_cols)
        self.feature_columns_ = cols_used

        y_arr = (np.asarray(y).ravel() > 0).astype(int)

        model_params = dict(self.params)
        if "min_child_samples" not in model_params and len(X_clean) < 40:
            model_params["min_child_samples"] = max(1, len(X_clean) // 4)

        self.model_ = build_classifier(
            params=model_params,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs,
        )

        clean_eval_set = None
        if eval_set:
            clean_eval_set = []
            for eval_X, eval_y in eval_set:
                clean_eval_X, _ = sanitize_features(eval_X, feature_columns=self.feature_columns_)
                clean_eval_y = (np.asarray(eval_y).ravel() > 0).astype(int)
                clean_eval_set.append((clean_eval_X, clean_eval_y))

        if clean_eval_set:
            self.model_.fit(X_clean, y_arr, eval_set=clean_eval_set, **fit_params)
        else:
            self.model_.fit(X_clean, y_arr, **fit_params)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities in [0.0, 1.0]."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' before predicting.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)

        X_clean, _ = sanitize_features(X, feature_columns=self.feature_columns_)
        raw_probs = self.model_.predict_proba(X_clean)

        if raw_probs.ndim == 2 and raw_probs.shape[1] >= 2:
            probs = raw_probs[:, 1]
        elif raw_probs.ndim == 2 and raw_probs.shape[1] == 1:
            probs = raw_probs[:, 0]
        else:
            probs = raw_probs.ravel()

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        return probs.astype(np.float32)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary match labels (0 or 1)."""
        t = self.threshold if threshold is None else float(threshold)
        probs = self.predict_proba(X)
        return predict_with_threshold(probs, threshold=t)

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray, Sequence],
        grid: Optional[Sequence[float]] = None,
        beta: float = 0.5,
    ) -> ThresholdSearchResult:
        """Search validation data to find threshold maximizing F_beta (default F0.5)."""
        val_probs = self.predict_proba(X_val)
        result = optimize_f05_threshold(y_val, val_probs, grid=grid, beta=beta)
        self.threshold = result.best_threshold
        self.best_threshold_ = result.best_threshold
        self.threshold_search_result_ = result
        return result

    def get_feature_importances(self, importance_type: str = "split") -> pd.Series:
        """Return feature importances as a pandas Series."""
        if self.model_ is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.model_.feature_importances_
        return pd.Series(importances, index=self.feature_columns_).sort_values(ascending=False)


def train_classifier(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray, Sequence],
    params: Optional[Dict[str, Any]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    class_weight: Optional[Union[str, Dict]] = "balanced",
    random_state: int = 42,
    **kwargs,
) -> EntityResolutionClassifier:
    """Train the LightGBM classifier on candidate pair similarity features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Pairwise features DataFrame.
    y_train : pd.Series or np.ndarray
        Binary labels (1 for true match, 0 for non-match).
    params : dict, optional
        Custom LightGBM parameters.
    feature_columns : list of str, optional
        Subset of features to train on. Defaults to FEATURE_COLUMNS if present.
    class_weight : str or dict, default='balanced'
        Weighting to address severe candidate pair imbalance.
    random_state : int, default=42
        Reproducibility seed.
    **kwargs :
        Additional arguments passed to model constructor or fit.

    Returns
    -------
    EntityResolutionClassifier
        Fitted classifier instance.
    """
    clf = EntityResolutionClassifier(
        params=params,
        feature_columns=feature_columns,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )
    clf.fit(X_train, y_train)
    return clf
