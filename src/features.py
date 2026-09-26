"""Feature engineering module for candidate pairs.

In the locked architecture:
Every candidate pair that survives the candidate union receives ALL similarity features.
Downstream, a single LightGBM classifier scores these feature vectors.
"""

from typing import List, Tuple
import pandas as pd


def compute_string_similarities(s1_text: str, s2_text: str) -> dict:
    """Compute string-level similarity metrics between two text records.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def extract_pair_features(
    s1_record: dict,
    candidate_record: dict,
    char_sim: float = 0.0,
    word_sim: float = 0.0,
    emb_sim: float = 0.0
) -> dict:
    """Compute the full set of similarity features for a candidate pair.
    
    Every candidate pair surviving the union receives ALL similarity features:
    - Name similarities (exact, token jaccard, edit distance, n-gram overlap, etc.)
    - Address similarities (token overlap, edit distance, component matching)
    - Country compatibility flag
    - Retrieval similarity scores (char TF-IDF, word TF-IDF, embedding cosine)
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    s1_lookup: dict,
    target_lookup: dict
) -> pd.DataFrame:
    """Build the complete feature matrix for all candidate pairs surviving the union.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")
