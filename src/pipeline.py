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
    train_mode: bool = True,
    config_override: Optional[dict] = None
) -> None:
    """Run the complete Business Entity Resolution pipeline.
    
    Generates:
    - output/candidate_pairs.tsv (blocking candidates fed to model)
    - output/matching_results.tsv (final match predictions)
    
    To be implemented during pipeline development.
    """
    raise NotImplementedError("To be implemented during pipeline development.")


if __name__ == "__main__":
    print("Locked Architecture Pipeline initialized.")
