"""
PyTorch Dataset
===============
Loads the Parquet files produced by Step 3 (training_dataset.py)
and serves batches to the training loop.

Design decisions:
    - Reads Parquet directly (faster than CSV, columnar format)
    - Normalizes continuous features at dataset level (not model level)
      so normalization stats can be saved and reused at inference time
    - Returns tensors already cast to correct dtype:
        LongTensor  for embedding indices (user_idx, item_idx)
        FloatTensor for continuous features and labels
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger
from typing import Optional
from dataclasses import dataclass, field


# ----------------------------------------------------------------
# FEATURE COLUMN DEFINITIONS
# Single source of truth — used by Dataset, Model, and FastAPI
# ----------------------------------------------------------------

# Integer index features (fed into Embedding layers)
USER_EMBEDDING_COLS = ["user_idx", "user_fav_cat_idx"]
ITEM_EMBEDDING_COLS = ["item_idx", "item_cat_idx"]

# Continuous features (fed into Dense layers after normalization)
USER_CONTINUOUS_COLS = [
    "user_click_count_7d_log",
    "user_purchase_count_30d",
    "user_total_spend_30d_log",
    "user_avg_engagement_score",
]

ITEM_CONTINUOUS_COLS = [
    "item_view_count_7d_log",
    "item_purchase_count_30d_log",
    "item_avg_rating_events",
    "item_cart_rate",
    "item_conversion_rate",
    "price_log",
    "avg_rating",
]

LABEL_COL = "label_binary"


# ----------------------------------------------------------------
# NORMALIZATION STATS
# Computed from training set, applied to val/test/serving
# ----------------------------------------------------------------

@dataclass
class NormalizationStats:
    """
    Mean and std for each continuous feature.
    Saved to disk so inference uses identical normalization.
    """
    means: dict[str, float] = field(default_factory=dict)
    stds:  dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Compute stats from training DataFrame."""
        for col in cols:
            if col in df.columns:
                self.means[col] = float(df[col].mean())
                self.stds[col]  = float(df[col].std())
                if self.stds[col] < 1e-8:
                    self.stds[col] = 1.0    # Avoid division by zero

    def transform(
        self,
        df: pd.DataFrame,
        cols: list[str]
    ) -> pd.DataFrame:
        """Apply z-score normalization using stored stats."""
        df = df.copy()
        for col in cols:
            if col in df.columns and col in self.means:
                df[col] = (df[col] - self.means[col]) / self.stds[col]
        return df

    def save(self, path: Path) -> None:
        """Persist stats to JSON for inference reuse."""
        with open(path, "w") as f:
            json.dump({"means": self.means, "stds": self.stds}, f, indent=2)
        logger.info(f"Normalization stats saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Load stats from JSON."""
        with open(path) as f:
            data = json.load(f)
        stats = cls()
        stats.means = data["means"]
        stats.stds  = data["stds"]
        logger.info(f"Normalization stats loaded ← {path}")
        return stats


# ----------------------------------------------------------------
# PYTORCH DATASET
# ----------------------------------------------------------------

class RecsysDataset(Dataset):
    """
    PyTorch Dataset for the Two-Tower recommendation model.

    Each sample contains:
        - User embedding indices  (LongTensor)
        - User continuous features (FloatTensor, normalized)
        - Item embedding indices  (LongTensor)
        - Item continuous features (FloatTensor, normalized)
        - Label                   (FloatTensor, binary 0/1)

    Args:
        parquet_path:   Path to train/val/test Parquet file
        norm_stats:     NormalizationStats instance (fit on train set)
        fit_norm:       If True, fit norm_stats on this dataset
                        (only True for training set)
    """

    def __init__(
        self,
        parquet_path: Path,
        norm_stats:   Optional[NormalizationStats] = None,
        fit_norm:     bool = False,
    ):
        self.parquet_path = Path(parquet_path)
        self.norm_stats   = norm_stats or NormalizationStats()

        logger.info(f"Loading dataset from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)
        logger.info(
            f"Dataset loaded | "
            f"rows={len(self.df):,} | "
            f"cols={len(self.df.columns)}"
        )

        # Fit normalization on training data
        all_continuous = USER_CONTINUOUS_COLS + ITEM_CONTINUOUS_COLS
        if fit_norm:
            self.norm_stats.fit(self.df, all_continuous)
            logger.info("Normalization stats fitted on training set")

        # Apply normalization
        self.df = self.norm_stats.transform(self.df, all_continuous)

        # Fill any remaining NaNs with 0
        self.df[all_continuous] = self.df[all_continuous].fillna(0.0)

        # Cache as numpy arrays for fast __getitem__
        self._cache_arrays()

        logger.success(
            f"✅ RecsysDataset ready | "
            f"positive_rate={self.df[LABEL_COL].mean():.2%}"
        )

    def _cache_arrays(self) -> None:
        """
        Pre-convert DataFrame columns to numpy arrays.
        This is 10-50x faster than accessing df.iloc[i] per sample.
        """
        self._user_emb   = self.df[USER_EMBEDDING_COLS].values.astype(np.int64)
        self._item_emb   = self.df[ITEM_EMBEDDING_COLS].values.astype(np.int64)
        self._user_cont  = self.df[USER_CONTINUOUS_COLS].values.astype(np.float32)
        self._item_cont  = self.df[ITEM_CONTINUOUS_COLS].values.astype(np.float32)
        self._labels     = self.df[LABEL_COL].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_emb_idx":  torch.tensor(self._user_emb[idx],  dtype=torch.long),
            "item_emb_idx":  torch.tensor(self._item_emb[idx],  dtype=torch.long),
            "user_features": torch.tensor(self._user_cont[idx],  dtype=torch.float32),
            "item_features": torch.tensor(self._item_cont[idx],  dtype=torch.float32),
            "label":         torch.tensor(self._labels[idx],     dtype=torch.float32),
        }


# ----------------------------------------------------------------
# DATALOADER FACTORY
# ----------------------------------------------------------------

def build_dataloaders(
    processed_dir: Path,
    batch_size:    int  = 2048,
    num_workers:   int  = 4,
    pin_memory:    bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, NormalizationStats]:
    """
    Build train, val, and test DataLoaders.

    Normalization is fitted ONLY on the training set
    and then applied identically to val/test/serving.

    Args:
        processed_dir:  Directory with train/val/test Parquet files
        batch_size:     Samples per batch
        num_workers:    Parallel data loading workers
        pin_memory:     Speed up GPU transfer (set False if no GPU)

    Returns:
        (train_loader, val_loader, test_loader, norm_stats)
    """
    processed_dir = Path(processed_dir)

    # Training set — fit normalization here
    norm_stats  = NormalizationStats()
    train_ds    = RecsysDataset(
        parquet_path = processed_dir / "train.parquet",
        norm_stats   = norm_stats,
        fit_norm     = True,   # ← Fit only on train
    )

    # Val and test — transform using train stats
    val_ds  = RecsysDataset(
        parquet_path = processed_dir / "val.parquet",
        norm_stats   = norm_stats,
        fit_norm     = False,  # ← Apply train stats
    )
    test_ds = RecsysDataset(
        parquet_path = processed_dir / "test.parquet",
        norm_stats   = norm_stats,
        fit_norm     = False,
    )

    # Save norm stats alongside processed data
    norm_stats.save(processed_dir / "norm_stats.json")

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,        # Shuffle training data each epoch
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,        # Drop incomplete last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,   # Larger batch for eval (no gradients)
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    logger.info(
        f"DataLoaders ready | "
        f"train={len(train_ds):,} | "
        f"val={len(val_ds):,} | "
        f"test={len(test_ds):,} | "
        f"batch_size={batch_size}"
    )

    return train_loader, val_loader, test_loader, norm_stats