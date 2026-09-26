"""Model training, threshold optimization, and inference module.

In the locked architecture:
- Exactly ONE downstream LightGBM classifier.
- Predictions scored against an F0.5 optimal decision threshold.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score (defaults to F_0.5)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, params: Optional[dict] = None):
    """Train the single downstream LightGBM classifier on extracted similarity features.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def optimize_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Find the optimal probability threshold maximizing macro F0.5 score.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def predict_matches(
    model,
    X_test: pd.DataFrame,
    threshold: float
) -> np.ndarray:
    """Generate binary match predictions using the calibrated F0.5 threshold.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")
