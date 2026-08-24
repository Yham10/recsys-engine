"""
Shared Path Configuration
==========================
THE single source of truth for all data paths.
All scripts import from here.

Directory structure enforced:
    recsys-engine/
    ├── data/
    │   ├── raw/              ← CSVs from data generator
    │   ├── processed/        ← Parquet files + encoders + norm_stats
    │   └── artifacts/        ← best_model.pt, item_embeddings.npy
    └── src/
        └── data_generator/
            └── data/raw/     ← Original CSVs (data generator writes here)
"""

from pathlib import Path

# ── Project root = recsys-engine/ ────────────────────────────────
# This file lives at recsys-engine/src/config_paths.py
# So parent = src/, parent.parent = recsys-engine/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Generator output (where CSVs are generated) ─────────────
DATA_GENERATOR_RAW_DIR = (
    PROJECT_ROOT / "src" / "data_generator" / "data" / "raw"
)

# ── Processed data (ONE location for all scripts) ─────────────────
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Model artifacts (ONE location for all scripts) ────────────────
DATA_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

# ── Feature repo path ─────────────────────────────────────────────
FEATURE_REPO_PATH = (
    PROJECT_ROOT / "src" / "feature_store" / "feature_repo"
)

# ── Auto-create directories on import ────────────────────────────
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Debug helper — print all resolved paths."""
    print(f"\n{'='*60}")
    print(f"  PROJECT_ROOT:            {PROJECT_ROOT}")
    print(f"  DATA_GENERATOR_RAW_DIR:  {DATA_GENERATOR_RAW_DIR}")
    print(f"  DATA_PROCESSED_DIR:      {DATA_PROCESSED_DIR}")
    print(f"  DATA_ARTIFACTS_DIR:      {DATA_ARTIFACTS_DIR}")
    print(f"  FEATURE_REPO_PATH:       {FEATURE_REPO_PATH}")
    print(f"{'='*60}")
    print(f"  RAW exists:       {DATA_GENERATOR_RAW_DIR.exists()}")
    print(f"  PROCESSED exists: {DATA_PROCESSED_DIR.exists()}")
    print(f"  ARTIFACTS exists: {DATA_ARTIFACTS_DIR.exists()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_paths()