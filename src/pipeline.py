"""End-to-end pipeline orchestration module.

Executes the locked architecture:
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
from typing import Optional


def run_pipeline(
    train_mode: bool = False,
    config_override: Optional[dict] = None
) -> dict:
    """Run the complete Business Entity Resolution pipeline.

    Generates:
    - output/candidate_pairs.tsv (blocking candidates fed to model)
    - output/matching_results.tsv (final match predictions)
    """
    if train_mode:
        from scripts.train_matcher import train_matcher
        override = config_override or {}
        return train_matcher(**override)
    else:
        from scripts.run_test_inference import run_test_inference
        override = config_override or {}
        return run_test_inference(**override)


if __name__ == "__main__":
    print("Locked Architecture Pipeline initialized.")
