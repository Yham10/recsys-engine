"""
Model Loader
============
Responsible for loading and managing:
    1. The Two-Tower PyTorch model from MLflow Registry
    2. The FAISS ANN index built from item embeddings
    3. Encoder mappings (user_id/item_id → integer indices)
    4. Normalization stats (for identical preprocessing to training)

This module is initialized ONCE at startup (lifespan event)
and shared across all requests via dependency injection.

Why FAISS?
    We have 5,000 items. At serving time, we need the top-K items
    most similar to the user embedding.
    Brute force = 5,000 dot products per request = fine for 5K items.
    FAISS IndexFlatIP = exact inner product search, no approximation needed.
    At 500K items, we'd switch to IndexIVFFlat (approximate but faster).
"""

import os
import sys
import json
import time
import numpy as np
import torch
import faiss
import mlflow
import mlflow.pytorch
from pathlib import Path
from loguru import logger
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent      # serving/
_SRC_DIR  = _THIS_DIR.parent                     # src/
sys.path.insert(0, str(_SRC_DIR))

from config_paths import DATA_PROCESSED_DIR, DATA_ARTIFACTS_DIR

# Import NormalizationStats — it lives in training/dataset.py
sys.path.insert(0, str(_SRC_DIR / "training"))
from dataset import NormalizationStats


# ----------------------------------------------------------------
# MODEL REGISTRY
# Singleton that holds all loaded artifacts
# ----------------------------------------------------------------

class ModelRegistry:
    """
    Holds all loaded ML artifacts needed for inference.

    Attributes:
        model:         Loaded TwoTowerModel (eval mode, on device)
        faiss_index:   FAISS index of item embeddings
        item_ids:      List mapping FAISS index → item_id string
        item_id_to_idx: Dict mapping item_id string → int index
        norm_stats:    Normalization stats from training
        encoders:      User/item/category encoder mappings
        model_version: String identifier of loaded model version
        device:        Torch device
        is_ready:      Whether all artifacts are loaded successfully
    """

    def __init__(self):
        self.model:           Optional[torch.nn.Module]     = None
        self.faiss_index:     Optional[faiss.Index]         = None
        self.item_ids:        Optional[list[str]]           = None
        self.item_id_to_idx:  Optional[dict[str, int]]      = None
        self.norm_stats:      Optional[NormalizationStats]  = None
        self.encoders:        Optional[dict]                = None
        self.model_version:   str                           = "unknown"
        self.device:          torch.device                  = torch.device("cpu")
        self.is_ready:        bool                          = False
        self._load_time:      float                         = 0.0


# Global singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Dependency injection hook for FastAPI routes."""
    return _registry


# ----------------------------------------------------------------
# LOADER FUNCTIONS
# ----------------------------------------------------------------

def load_model_from_mlflow(settings) -> tuple:
    """Load latest model version from MLflow Registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri
        )

        # Search all versions — avoid deprecated stage API
        all_versions = client.search_model_versions(
            f"name='{settings.mlflow_model_name}'"
        )

        if not all_versions:
            raise ValueError(
                f"No versions found for model '{settings.mlflow_model_name}'"
            )

        # Take the highest version number
        latest = sorted(
            all_versions,
            key=lambda v: int(v.version),
            reverse=True
        )[0]

        version_num = latest.version
        run_id      = latest.run_id
        model_uri   = f"runs:/{run_id}/model"

        logger.info(
            f"Loading model from MLflow | "
            f"name={settings.mlflow_model_name} | "
            f"version={version_num}"
        )

        model = mlflow.pytorch.load_model(
            model_uri    = model_uri,
            map_location = "cpu",
        )
        model.eval()

        logger.success(
            f"✅ Model loaded from MLflow | version={version_num}"
        )
        return model, f"mlflow-v{version_num}"

    except Exception as e:
        logger.warning(
            f"MLflow load failed: {e}\n"
            f"Using local checkpoint fallback..."
        )
        return _load_model_local_fallback(settings)


def _load_model_local_fallback(settings) -> tuple:
    """Load from local best_model.pt checkpoint."""
    checkpoint_path = DATA_ARTIFACTS_DIR / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No model found at {checkpoint_path.resolve()}\n"
            f"Run training first: cd src/training && python run_training.py"
        )

    # Import model from training/
    _SRC_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_SRC_DIR / "training"))
    from model import TwoTowerModel, ModelConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config     = ModelConfig(**checkpoint["config"])
    model      = TwoTowerModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.success(
        f"✅ Model loaded from local checkpoint | "
        f"path={checkpoint_path.resolve()}"
    )
    return model, "local-checkpoint"


def _load_model_local_fallback(settings) -> tuple:
    """
    Fallback: load model from local checkpoint file.
    Used when MLflow is unavailable or in testing.
    """
    checkpoint_path = settings.artifacts_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No model found at {checkpoint_path}.\n"
            f"Run Step 4 training first."
        )

    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
    from model import TwoTowerModel, ModelConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config     = ModelConfig(**checkpoint["config"])
    model      = TwoTowerModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.success("✅ Model loaded from local checkpoint (fallback)")
    return model, "local-checkpoint"


def build_faiss_index(
    embeddings: np.ndarray,
    n_probe:    int = 10,
) -> faiss.Index:
    """
    Build a FAISS inner product index from item embeddings.

    Index type: IndexFlatIP (exact inner product / cosine similarity)
    For n_items < 100K, exact search is fast enough (<5ms).
    For larger catalogs, use IndexIVFFlat (approximate).

    Args:
        embeddings: Item embeddings [n_items, embedding_dim]
        n_probe:    Number of clusters to probe (IVF only)

    Returns:
        Trained FAISS index
    """
    n_items, dim = embeddings.shape
    logger.info(
        f"Building FAISS index | "
        f"n_items={n_items:,} | dim={dim} | type=IndexFlatIP"
    )

    # Normalize embeddings for cosine similarity search
    faiss.normalize_L2(embeddings)

    # Use flat (exact) index for our catalog size
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)   # Wrap to store custom IDs

    # Add embeddings with integer IDs
    ids = np.arange(n_items, dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    logger.success(
        f"✅ FAISS index built | "
        f"total_vectors={index.ntotal:,}"
    )
    return index


def load_all_artifacts(settings) -> None:
    """
    Master loading function called at application startup.
    Uses absolute paths from config_paths — no ambiguity.
    """
    global _registry
    start = time.time()

    # Use absolute paths — ignore settings paths entirely
    processed_dir = DATA_PROCESSED_DIR
    artifacts_dir = DATA_ARTIFACTS_DIR

    logger.info("=" * 60)
    logger.info("  LOADING ML ARTIFACTS")
    logger.info(f"  processed_dir = {processed_dir.resolve()}")
    logger.info(f"  artifacts_dir = {artifacts_dir.resolve()}")
    logger.info("=" * 60)

    # ── 1. Load encoders ──────────────────────────────────────────
    encoder_path = processed_dir / "encoders.json"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"encoders.json not found at {encoder_path.resolve()}\n"
            f"Run: cd src/feature_store && python training_dataset.py"
        )
    with open(encoder_path) as f:
        _registry.encoders = json.load(f)

    _registry.item_ids = _registry.encoders["item_classes"]
    _registry.item_id_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(_registry.item_ids)
    }
    logger.info(
        f"Encoders loaded | "
        f"users={_registry.encoders['n_users']:,} | "
        f"items={len(_registry.item_ids):,} | "
        f"categories={_registry.encoders['n_categories']}"
    )

    # ── 2. Load normalization stats ───────────────────────────────
    norm_path = processed_dir / "norm_stats.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"norm_stats.json not found at {norm_path.resolve()}\n"
            f"Run training first: cd src/training && python run_training.py"
        )
    _registry.norm_stats = NormalizationStats.load(norm_path)
    logger.info("Normalization stats loaded")

    # ── 3. Load model ─────────────────────────────────────────────
    model, version = load_model_from_mlflow(settings)
    _registry.model         = model
    _registry.model_version = version

    # ── 4. Load item embeddings → FAISS ──────────────────────────
    emb_path = artifacts_dir / "item_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"item_embeddings.npy not found at {emb_path.resolve()}\n"
            f"Run training first: cd src/training && python run_training.py"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    _registry.faiss_index = build_faiss_index(
        embeddings, n_probe=settings.faiss_n_probe
    )

    # ── 5. Device ────────────────────────────────────────────────
    if torch.cuda.is_available():
        _registry.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _registry.device = torch.device("mps")
    else:
        _registry.device = torch.device("cpu")

    _registry.model = _registry.model.to(_registry.device)

    # ── Mark ready ───────────────────────────────────────────────
    _registry.is_ready   = True
    _registry._load_time = time.time() - start

    logger.info("=" * 60)
    logger.info("  ALL ARTIFACTS LOADED")
    logger.info(f"  Model version:  {_registry.model_version}")
    logger.info(f"  Device:         {_registry.device}")
    logger.info(f"  FAISS vectors:  {_registry.faiss_index.ntotal:,}")
    logger.info(f"  n_categories:   {_registry.encoders['n_categories']}")
    logger.info(f"  Load time:      {_registry._load_time:.2f}s")
    logger.info("=" * 60)

def unload_artifacts() -> None:
    """
    Clean up loaded artifacts on shutdown.
    Releases GPU memory and file handles.
    """
    global _registry
    _registry.model        = None
    _registry.faiss_index  = None
    _registry.is_ready     = False
    logger.info("ML artifacts unloaded")