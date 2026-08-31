"""
Training Dataset Builder
========================
Builds the final point-in-time correct training dataset.

Saves ALL outputs to:
    recsys-engine/data/processed/
        train.parquet
        val.parquet
        test.parquet
        encoders.json
        norm_stats.json  (saved later by dataset.py during training)
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Path setup ───────────────────────────────────────────────────
# This file: recsys-engine/src/feature_store/training_dataset.py
# We need:   recsys-engine/src/  on the path
_THIS_DIR = Path(__file__).resolve().parent          # feature_store/
_SRC_DIR  = _THIS_DIR.parent                         # src/
sys.path.insert(0, str(_SRC_DIR))

from config_paths import (
    DATA_GENERATOR_RAW_DIR,
    DATA_PROCESSED_DIR,
    print_paths,
)
from feast_manager import FeastManager

# ── Logging ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ── Fixed category vocabulary ─────────────────────────────────────
# LabelEncoder sorts alphabetically — this matches exactly
ALL_CATEGORIES = sorted([
    "automotive", "beauty",   "books",
    "clothing",   "electronics", "food",
    "home_garden", "jewelry", "sports", "toys",
])   # 10 categories


class TrainingDatasetBuilder:
    """
    Orchestrates construction of a training-ready dataset.

    Pipeline:
        1. Load raw interactions (entity_df)
        2. Feast point-in-time correct feature join
        3. Preprocess & encode
        4. Temporal train/val/test split
        5. Save Parquet to recsys-engine/data/processed/
    """

    def __init__(
        self,
        interactions_path: Path = DATA_GENERATOR_RAW_DIR / "interactions.csv",
        output_dir:        Path = DATA_PROCESSED_DIR,
        sample_size:       int  = None,
    ):
        self.interactions_path = Path(interactions_path)
        self.output_dir        = Path(output_dir)
        self.sample_size       = sample_size
        self.manager           = FeastManager()

        # Encoders
        self.user_encoder     = LabelEncoder()
        self.item_encoder     = LabelEncoder()
        self.category_encoder = LabelEncoder()

        # Pre-fit category encoder on ALL known categories
        self.category_encoder.fit(ALL_CATEGORIES)

        logger.info(
            f"TrainingDatasetBuilder initialized | "
            f"output_dir={self.output_dir.resolve()} | "
            f"n_categories={len(ALL_CATEGORIES)}"
        )

    def build(self) -> dict[str, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("  TRAINING DATASET BUILD STARTED")
        logger.info(f"  Interactions: {self.interactions_path.resolve()}")
        logger.info(f"  Output:       {self.output_dir.resolve()}")
        logger.info("=" * 60)

        entity_df   = self.manager.build_entity_dataframe(
            interactions_path = str(self.interactions_path),
            sample_size       = self.sample_size,
        )
        training_df = self.manager.get_training_dataset(entity_df)
        training_df = self._preprocess(training_df)
        self._validate(training_df)
        splits      = self._split(training_df)
        self._save(splits)
        self._summarize(splits)
        return splits

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing dataset | shape={df.shape}")
        df = df.copy()

        # ── Fill missing values ───────────────────────────────────
        num_fill = {
            "user_click_count_7d":       0,
            "user_purchase_count_30d":   0,
            "user_total_spend_30d":      0.0,
            "user_avg_engagement_score": 0.3,
            "item_view_count_7d":        0,
            "item_purchase_count_30d":   0,
            "item_avg_rating_events":    3.5,
            "item_cart_rate":            0.0,
            "item_conversion_rate":      0.0,
            "price":                     0.0,
            "avg_rating":                3.5,
        }
        df = df.fillna(num_fill)
        df["user_favorite_category"] = (
            df["user_favorite_category"].fillna("electronics")
        )
        df["category"] = df["category"].fillna("electronics")

        # ── Encode user/item IDs → integers ──────────────────────
        df["user_idx"] = self.user_encoder.fit_transform(df["user_id"])
        df["item_idx"] = self.item_encoder.fit_transform(df["item_id"])

        # ── Encode categories using FIXED vocabulary ──────────────
        # Map unknowns to "electronics" before encoding
        df["user_favorite_category"] = df["user_favorite_category"].apply(
            lambda x: x if x in ALL_CATEGORIES else "electronics"
        )
        df["category"] = df["category"].apply(
            lambda x: x if x in ALL_CATEGORIES else "electronics"
        )

        df["user_fav_cat_idx"] = self.category_encoder.transform(
            df["user_favorite_category"]
        )
        df["item_cat_idx"] = self.category_encoder.transform(
            df["category"]
        )

        # ── Clip outliers ─────────────────────────────────────────
        df["user_click_count_7d"]     = df["user_click_count_7d"].clip(0, 500)
        df["user_purchase_count_30d"] = df["user_purchase_count_30d"].clip(0, 100)
        df["user_total_spend_30d"]    = df["user_total_spend_30d"].clip(0, 50_000)
        df["item_view_count_7d"]      = df["item_view_count_7d"].clip(0, 10_000)
        df["price"]                   = df["price"].clip(0, 10_000)

        # ── Log-transform skewed features ─────────────────────────
        df["user_click_count_7d_log"]     = np.log1p(df["user_click_count_7d"])
        df["user_total_spend_30d_log"]    = np.log1p(df["user_total_spend_30d"])
        df["item_view_count_7d_log"]      = np.log1p(df["item_view_count_7d"])
        df["item_purchase_count_30d_log"] = np.log1p(df["item_purchase_count_30d"])
        df["price_log"]                   = np.log1p(df["price"])

        # ── Binarize label ────────────────────────────────────────
        df["label_binary"] = (df["label"] >= 0.5).astype(int)

        logger.info(
            f"Preprocessing complete | "
            f"shape={df.shape} | "
            f"positive_rate={df['label_binary'].mean():.2%} | "
            f"n_categories={len(self.category_encoder.classes_)}"
        )
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        logger.info("Validating dataset quality...")
        checks = {
            "No empty dataset":           len(df) > 0,
            "Has user_idx column":        "user_idx" in df.columns,
            "Has item_idx column":        "item_idx" in df.columns,
            "Has label column":           "label" in df.columns,
            "No all-null rows":           df.isnull().all(axis=1).sum() == 0,
            "Positive label rate > 1%":   df["label_binary"].mean() > 0.01,
            "Positive label rate < 99%":  df["label_binary"].mean() < 0.99,
            "Price non-negative":         (df["price"] >= 0).all(),
            "Ratings in valid range":     df["avg_rating"].between(0, 5).all(),
            "n_categories == 10":         len(self.category_encoder.classes_) == 10,
        }
        failed = []
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check_name}")
            if not result:
                failed.append(check_name)

        if failed:
            raise ValueError(f"Dataset validation FAILED: {failed}")

        logger.success("✅ All dataset quality checks passed")

    def _split(
        self,
        df:          pd.DataFrame,
        train_ratio: float = 0.80,
        val_ratio:   float = 0.10,
    ) -> dict[str, pd.DataFrame]:
        logger.info("Splitting dataset temporally...")
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        n  = len(df)

        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()

        logger.info(
            f"Split sizes → "
            f"train={len(train):,} | val={len(val):,} | test={len(test):,}"
        )
        return {"train": train, "val": val, "test": test}

    def _save(self, splits: dict[str, pd.DataFrame]) -> None:
        """Save splits + encoder mappings to DATA_PROCESSED_DIR."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            path = self.output_dir / f"{split_name}.parquet"
            split_df.to_parquet(path, index=False, compression="snappy")
            logger.success(
                f"✅ Saved {split_name} → {path.resolve()} "
                f"({len(split_df):,} rows)"
            )

        # ── Save encoders.json ────────────────────────────────────
        encoders = {
            "n_users":      len(self.user_encoder.classes_),
            "n_items":      len(self.item_encoder.classes_),
            "n_categories": len(self.category_encoder.classes_),  # Always 10
            "user_classes": list(self.user_encoder.classes_),
            "item_classes": list(self.item_encoder.classes_),
            "cat_classes":  list(self.category_encoder.classes_),
        }

        encoder_path = self.output_dir / "encoders.json"
        with open(encoder_path, "w") as f:
            json.dump(encoders, f, indent=2)

        logger.success(
            f"✅ Encoder mappings saved → {encoder_path.resolve()} | "
            f"n_categories={encoders['n_categories']}"
        )

    def _summarize(self, splits: dict[str, pd.DataFrame]) -> None:
        train = splits["train"]
        logger.info("=" * 60)
        logger.info("  DATASET BUILD COMPLETE")
        logger.info(f"  Output dir:    {self.output_dir.resolve()}")
        logger.info(f"  Total rows:    {sum(len(v) for v in splits.values()):>10,}")
        logger.info(f"  Train rows:    {len(splits['train']):>10,}")
        logger.info(f"  Val rows:      {len(splits['val']):>10,}")
        logger.info(f"  Test rows:     {len(splits['test']):>10,}")
        logger.info(f"  Unique users:  {train['user_id'].nunique():>10,}")
        logger.info(f"  Unique items:  {train['item_id'].nunique():>10,}")
        logger.info(f"  n_categories:  {len(self.category_encoder.classes_):>10}")
        logger.info(f"  Positive rate: {train['label_binary'].mean():>10.2%}")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time correct training dataset from Feast"
    )
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    args    = parse_args()
    builder = TrainingDatasetBuilder(sample_size=args.sample_size)
    builder.build()


if __name__ == "__main__":
    main()