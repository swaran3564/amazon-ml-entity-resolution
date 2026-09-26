"""Configuration settings for the Business Entity Resolution pipeline.

Locked Architecture:
S1
│
├── Character TF-IDF retrieval
├── Word TF-IDF retrieval
└── Embedding retrieval
          │
          ▼
    Candidate UNION
          │
          ▼
    ALL similarity features
          │
          ▼
       LightGBM
          │
          ▼
    F0.5 threshold
          │
          ▼
    Final matches
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# Dataset Files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE = OUTPUT_DIR / "candidate_pairs.tsv"

# Candidate Retrieval Configuration (Blocking)
CHAR_TFIDF_TOP_K = 20
CHAR_NGRAM_RANGE = (2, 4)

WORD_TFIDF_TOP_K = 20
WORD_NGRAM_RANGE = (1, 2)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_TOP_K = 20
EMBEDDING_BATCH_SIZE = 256

# Downstream LightGBM Classifier Parameters
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

# Decision Threshold Optimization for F0.5
BETA_SCORE = 0.5
DEFAULT_THRESHOLD = 0.5
