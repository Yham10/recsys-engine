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

import sys

from config import get_settings

# Allow imports from src/training
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

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

def load_model_from_mlflow(settings) -> torch.nn.Module:
    """
    Load the latest production model from MLflow Model Registry.

    Falls back to loading from local artifacts if MLflow is unavailable.
    This makes the service resilient to MLflow downtime.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = (
        f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_stage}"
    )

    logger.info(
        f"Loading model from MLflow | "
        f"uri={model_uri}"
    )

    try:
        model = mlflow.pytorch.load_model(
            model_uri  = model_uri,
            map_location = "cpu",
        )
        model.eval()

        # Get model version metadata
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            settings.mlflow_model_name,
            stages=[settings.mlflow_model_stage]
        )
        version_str = versions[0].version if versions else "unknown"
        logger.success(
            f"✅ Model loaded from MLflow | version={version_str}"
        )
        return model, version_str

    except Exception as e:
        logger.warning(
            f"MLflow model load failed: {e}\n"
            f"Attempting fallback to local artifacts..."
        )
        return _load_model_local_fallback(settings)


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
    Loads all artifacts into the global ModelRegistry singleton.

    Raises:
        RuntimeError if any critical artifact fails to load.
    """
    global _registry
    start = time.time()

    logger.info("=" * 60)
    logger.info("  LOADING ML ARTIFACTS")
    logger.info("=" * 60)

    # ---- 1. Load encoders ----
    encoder_path = settings.processed_dir / "encoders.json"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"encoders.json not found at {encoder_path}. Run Step 3."
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
        f"items={len(_registry.item_ids):,}"
    )

    # ---- 2. Load normalization stats ----
    norm_path = settings.processed_dir / "norm_stats.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"norm_stats.json not found at {norm_path}. Run Step 3."
        )
    _registry.norm_stats = NormalizationStats.load(norm_path)
    logger.info("Normalization stats loaded")

    # ---- 3. Load model ----
    model, version = load_model_from_mlflow(settings)
    _registry.model         = model
    _registry.model_version = version

    # ---- 4. Load item embeddings and build FAISS index ----
    emb_path = settings.artifacts_dir / "item_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"item_embeddings.npy not found at {emb_path}. Run Step 4."
        )

    embeddings = np.load(emb_path).astype(np.float32)
    _registry.faiss_index = build_faiss_index(
        embeddings, n_probe=settings.faiss_n_probe
    )

    # ---- 5. Select device ----
    if torch.cuda.is_available():
        _registry.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _registry.device = torch.device("mps")
    else:
        _registry.device = torch.device("cpu")

    _registry.model = _registry.model.to(_registry.device)

    # ---- Mark ready ----
    _registry.is_ready  = True
    _registry._load_time = time.time() - start

    logger.info("=" * 60)
    logger.info("  ALL ARTIFACTS LOADED")
    logger.info(f"  Model version:  {_registry.model_version}")
    logger.info(f"  Device:         {_registry.device}")
    logger.info(f"  FAISS vectors:  {_registry.faiss_index.ntotal:,}")
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