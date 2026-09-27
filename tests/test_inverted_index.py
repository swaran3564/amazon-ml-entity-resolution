import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.inverted_index import (
    normalize_country,
    retrieve_candidates_from_index,
    build_or_load_country_inverted_index,
)


def test_normalize_country():
    assert normalize_country(" US ") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("FRANCE") == "france"
    assert normalize_country(None) == ""
    assert normalize_country(np.nan) == ""
    assert normalize_country("") == ""
    assert normalize_country("   ") == ""
    assert normalize_country("Germany") == "germany"


def test_retrieve_candidates_from_index_basic():
    # Build toy vectorizers
    corpus = ["apple inc", "banana corp", "apple banana store", "orange tech"]
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit(corpus)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1)).fit(corpus)

    # Targets in India: apple inc (T1), banana corp (T2)
    # Targets in US: apple banana store (T3), orange tech (T4)
    t_in = ["apple inc", "banana corp"]
    t_in_ids = np.array(["T1", "T2"])
    c_char_in = char_vec.transform(t_in).T.tocsr()
    c_word_in = word_vec.transform(t_in).T.tocsr()

    t_us = ["apple banana store", "orange tech"]
    t_us_ids = np.array(["T3", "T4"])
    c_char_us = char_vec.transform(t_us).T.tocsr()
    c_word_us = word_vec.transform(t_us).T.tocsr()

    country_indexes = {
        "india": (c_char_in, c_word_in, t_in_ids),
        "us": (c_char_us, c_word_us, t_us_ids),
    }

    s1_df = pd.DataFrame([
        {"entity_id": "S1", "business_name": "apple store", "country": "india"},
        {"entity_id": "S2", "business_name": "orange corp", "country": "us"},
        {"entity_id": "S3", "business_name": "banana market", "country": ""},  # fallback query
    ])

    cands_map, char_scores, word_scores = retrieve_candidates_from_index(
        s1_df=s1_df,
        country_indexes=country_indexes,
        char_vec=char_vec,
        word_vec=word_vec,
        char_top_k=5,
        word_top_k=5,
    )

    # S1 is India: should only match from India (T1 or T2), not US
    assert "S1" in cands_map
    assert "T1" in cands_map["S1"]
    assert "T3" not in cands_map["S1"]
    assert "T4" not in cands_map["S1"]

    # S2 is US: should match T4 (orange tech), not India
    assert "S2" in cands_map
    assert "T4" in cands_map["S2"]
    assert "T1" not in cands_map["S2"]

    # S3 has empty country: fallback searches all countries
    assert "S3" in cands_map
    assert any(cid in cands_map["S3"] for cid in ["T2", "T3"])


def test_inverted_index_caching(tmp_path):
    # Create small mock S2 and S3 files
    s2_path = tmp_path / "s2.tsv"
    s3_path = tmp_path / "s3.tsv"

    s2_df = pd.DataFrame([
        {"entity_id": "S2_1", "business_name": "Alpha Corp", "country": "India"},
        {"entity_id": "S2_2", "business_name": "Beta LLC", "country": "US"},
    ])
    s2_df.to_csv(s2_path, sep="\t", index=False)

    s3_df = pd.DataFrame([
        {"entity_id": "S3_1", "business_name": "Gamma Inc", "country": "India"},
    ])
    s3_df.to_csv(s3_path, sep="\t", index=False)

    corpus = ["Alpha Corp", "Beta LLC", "Gamma Inc"]
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit(corpus)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1)).fit(corpus)

    cache_dir = tmp_path / "cache"

    # 1. Build from scratch
    indexes1 = build_or_load_country_inverted_index(
        split_name="toy",
        s2_path=s2_path,
        s3_path=s3_path,
        char_vec=char_vec,
        word_vec=word_vec,
        cache_dir=cache_dir,
        chunk_size=10,
    )
    assert "india" in indexes1
    assert "us" in indexes1
    assert len(indexes1["india"][2]) == 2  # S2_1 and S3_1
    assert len(indexes1["us"][2]) == 1     # S2_2

    indexes2 = build_or_load_country_inverted_index(
        split_name="toy",
        s2_path=s2_path,
        s3_path=s3_path,
        char_vec=char_vec,
        word_vec=word_vec,
        cache_dir=cache_dir,
        chunk_size=10,
    )
    assert set(indexes1.keys()) == set(indexes2.keys())
    assert np.array_equal(indexes1["india"][2], indexes2["india"][2])


def test_stop_ngram_extraction_and_pruning():
    from src.inverted_index import extract_corpus_stop_ngrams, load_stop_ngrams_config

    corpus = [
        "acme private limited",
        "beta private limited",
        "gamma private limited",
        "delta technologies",
    ]
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3)).fit(corpus)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1)).fit(corpus)

    t_in_ids = np.array(["T1", "T2", "T3", "T4"])
    c_char = char_vec.transform(corpus).T.tocsr()
    c_word = word_vec.transform(corpus).T.tocsr()

    country_indexes = {"india": (c_char, c_word, t_in_ids)}

    # Extract stop n-grams appearing in > 2 targets (e.g. "private", "limited")
    stops = extract_corpus_stop_ngrams(country_indexes, char_vec, word_vec, df_threshold=2)
    assert "india" in stops
    assert "limited" in stops["india"]["word"]
    assert "private" in stops["india"]["word"]

    # Test retrieval with pruning: query "acme private limited"
    # Should still retrieve T1 (acme) because "acme" is non-stop
    s1_df = pd.DataFrame([{"entity_id": "S1", "business_name": "acme private limited", "country": "india"}])
    cands, c_sc, w_sc = retrieve_candidates_from_index(
        s1_df=s1_df,
        country_indexes=country_indexes,
        char_vec=char_vec,
        word_vec=word_vec,
        char_top_k=2,
        word_top_k=2,
        stop_ngrams=stops,
        use_stop_ngrams=True,
    )
    assert "S1" in cands
    assert "T1" in cands["S1"]

    # Test all-stop query fallback: query has ONLY stop words ("private limited")
    # Should NOT return empty candidates because of fallback logic
    s1_stop_only = pd.DataFrame([{"entity_id": "S2", "business_name": "private limited", "country": "india"}])
    cands_stop, _, _ = retrieve_candidates_from_index(
        s1_df=s1_stop_only,
        country_indexes=country_indexes,
        char_vec=char_vec,
        word_vec=word_vec,
        char_top_k=2,
        word_top_k=2,
        stop_ngrams=stops,
        use_stop_ngrams=True,
    )
    assert "S2" in cands_stop
    assert len(cands_stop["S2"]) > 0

