"""Candidate generation / blocking module.

Locked candidate-generation architecture:
1. Character TF-IDF retrieval
2. Word TF-IDF retrieval
3. Embedding retrieval (from src/embeddings.py)
Candidate UNION: Combines candidates from all three retrieval channels.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd


def retrieve_char_tfidf_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 20
) -> Dict[str, Set[str]]:
    """Retrieve candidate target IDs for each S1 entity using Character n-gram TF-IDF similarity.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def retrieve_word_tfidf_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 20
) -> Dict[str, Set[str]]:
    """Retrieve candidate target IDs for each S1 entity using Word n-gram TF-IDF similarity.
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


def union_candidates(
    char_candidates: Dict[str, Set[str]],
    word_candidates: Dict[str, Set[str]],
    embedding_candidates: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """Form the Candidate UNION across Character TF-IDF, Word TF-IDF, and Embedding retrievals.
    
    Returns a mapping of source1_entity_id -> set of candidate entity_ids.
    """
    all_s1_ids = set(char_candidates) | set(word_candidates) | set(embedding_candidates)
    combined: Dict[str, Set[str]] = {}
    for s1_id in all_s1_ids:
        c_set = set()
        c_set.update(char_candidates.get(s1_id, set()))
        c_set.update(word_candidates.get(s1_id, set()))
        c_set.update(embedding_candidates.get(s1_id, set()))
        combined[s1_id] = c_set
    return combined
