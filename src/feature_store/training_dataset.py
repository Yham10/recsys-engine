"""
Training Dataset Builder
========================
Builds the final point-in-time correct training dataset
by combining raw interactions with Feast historical features.

This is the script that the Airflow training DAG calls
before launching model training.

Output: A single Parquet file ready for the PyTorch training loop.

Point-in-Time Correctness — Why it matters:
    Imagine user_000001 bought a product on Jan 15.
    On Jan 14, they had clicked 5 items in the last 7 days.
    On Jan 20, they had clicked 12 items in the last 7 days.

    A naive system would use Jan 20 features to label the Jan 15 purchase.
    That's DATA LEAKAGE — the model sees the future during training.

    Feast's point-in-time join uses Jan 14 features for the Jan 15 row.
    THIS is what makes our model trustworthy in production.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from feast_manager import FeastManager


# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

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

DATA_RAW_DIR = Path(__file__).parent.parent / "data_generator" / "data" / "raw"
DATA_PROCESSED_DIR = Path("data/processed")
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# TRAINING DATASET BUILDER
# ----------------------------------------------------------------

class TrainingDatasetBuilder:
    """
    Orchestrates the construction of a training-ready dataset.

    Pipeline:
        1. Load raw interactions (entity_df)
        2. Call Feast for point-in-time correct features
        3. Preprocess & encode categorical features
        4. Split into train / validation / test sets
        5. Save to Parquet for efficient loading in PyTorch
    """

    def __init__(
        self,
        interactions_path: Path = DATA_RAW_DIR / "interactions.csv",
        output_dir:        Path = DATA_PROCESSED_DIR,
        sample_size:       int  = None,
    ):
        self.interactions_path = interactions_path
        self.output_dir        = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size       = sample_size
        self.manager           = FeastManager()

        # Label encoders — saved for use at inference time
        self.user_encoder     = LabelEncoder()
        self.item_encoder     = LabelEncoder()
        self.category_encoder = LabelEncoder()

        logger.info("TrainingDatasetBuilder initialized")

    def build(self) -> dict[str, pd.DataFrame]:
        """
        Full pipeline: interactions → Feast join → preprocess → split → save.

        Returns:
            Dict with keys: 'train', 'val', 'test'
        """
        logger.info("=" * 60)
        logger.info("  TRAINING DATASET BUILD STARTED")
        logger.info("=" * 60)

        # Step 1: Build entity DataFrame
        entity_df = self.manager.build_entity_dataframe(
            interactions_path = str(self.interactions_path),
            sample_size       = self.sample_size,
        )

        # Step 2: Retrieve point-in-time correct features from Feast
        training_df = self.manager.get_training_dataset(entity_df)

        # Step 3: Preprocess features
        training_df = self._preprocess(training_df)

        # Step 4: Validate dataset quality
        self._validate(training_df)

        # Step 5: Split into train/val/test
        splits = self._split(training_df)

        # Step 6: Save to Parquet
        self._save(splits)

        # Step 7: Print dataset summary
        self._summarize(splits)

        return splits

    # ----------------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and encode features for model consumption.

        Operations:
            - Fill NaN values with sensible defaults
            - Encode user_id, item_id as integers (for embedding layers)
            - Encode categorical features
            - Clip outliers (spend, click counts)
            - Add interaction-level features
        """
        logger.info(f"Preprocessing dataset | shape={df.shape}")
        df = df.copy()

        # ---- Fill missing values ----
        num_fill = {
            "user_click_count_7d":       0,
            "user_purchase_count_30d":   0,
            "user_total_spend_30d":      0.0,
            "user_avg_engagement_score": 0.3,
            "item_view_count_7d":        0,
            "item_purchase_count_30d":   0,
            "item_avg_rating_events":    3.5,   # Population average
            "item_cart_rate":            0.0,
            "item_conversion_rate":      0.0,
            "price":                     0.0,
            "avg_rating":                3.5,
        }
        df.fillna(num_fill, inplace=True)
        df["user_favorite_category"].fillna("electronics", inplace=True)
        df["category"].fillna("electronics", inplace=True)

        # ---- Encode entity IDs as integers ----
        # PyTorch Embedding layers need integer indices, not strings
        df["user_idx"] = self.user_encoder.fit_transform(df["user_id"])
        df["item_idx"] = self.item_encoder.fit_transform(df["item_id"])

        # ---- Encode categorical features ----
        df["user_fav_cat_idx"] = self.category_encoder.fit_transform(
            df["user_favorite_category"]
        )
        # Item category shares the same encoder as user favorite category
        df["item_cat_idx"] = self.category_encoder.transform(
            df["category"].map(
                lambda x: x if x in self.category_encoder.classes_ else "electronics"
            )
        )

        # ---- Clip outliers ----
        df["user_click_count_7d"]     = df["user_click_count_7d"].clip(0, 500)
        df["user_purchase_count_30d"] = df["user_purchase_count_30d"].clip(0, 100)
        df["user_total_spend_30d"]    = df["user_total_spend_30d"].clip(0, 50_000)
        df["item_view_count_7d"]      = df["item_view_count_7d"].clip(0, 10_000)
        df["price"]                   = df["price"].clip(0, 10_000)

        # ---- Log-transform skewed features ----
        df["user_click_count_7d_log"]     = np.log1p(df["user_click_count_7d"])
        df["user_total_spend_30d_log"]    = np.log1p(df["user_total_spend_30d"])
        df["item_view_count_7d_log"]      = np.log1p(df["item_view_count_7d"])
        df["item_purchase_count_30d_log"] = np.log1p(df["item_purchase_count_30d"])
        df["price_log"]                   = np.log1p(df["price"])

        # ---- Binarize label (implicit feedback) ----
        # Label = 1 if strong positive signal (purchase/cart/wishlist)
        # Label = 0 for weaker signals (views, page views)
        df["label_binary"] = (df["label"] >= 0.5).astype(int)

        logger.info(
            f"Preprocessing complete | "
            f"shape={df.shape} | "
            f"positive_rate={df['label_binary'].mean():.2%}"
        )
        return df

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Assert dataset quality standards.
        Raises ValueError if any check fails.
        This prevents corrupted data from reaching the model.
        """
        logger.info("Validating dataset quality...")

        checks = {
            "No empty dataset":
                len(df) > 0,
            "Has user_idx column":
                "user_idx" in df.columns,
            "Has item_idx column":
                "item_idx" in df.columns,
            "Has label column":
                "label" in df.columns,
            "No all-null rows":
                df.isnull().all(axis=1).sum() == 0,
            "Positive label rate > 1%":
                df["label_binary"].mean() > 0.01,
            "Positive label rate < 99%":
                df["label_binary"].mean() < 0.99,
            "Price values are non-negative":
                (df["price"] >= 0).all(),
            "Ratings are in valid range":
                df["avg_rating"].between(0, 5).all(),
        }

        failed = []
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check_name}")
            if not result:
                failed.append(check_name)

        if failed:
            raise ValueError(
                f"Dataset validation FAILED. Failed checks: {failed}"
            )

        logger.success("✅ All dataset quality checks passed")

    # ----------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # ----------------------------------------------------------

    def _split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.80,
        val_ratio:   float = 0.10,
        test_ratio:  float = 0.10,
    ) -> dict[str, pd.DataFrame]:
        """
        Temporal train/val/test split.

        IMPORTANT: We split by TIME, not randomly.
        Random splitting would leak future interactions into training.
        By splitting temporally, we simulate real deployment conditions.

        Timeline: ──[train 80%]──[val 10%]──[test 10%]──► time
        """
        logger.info(
            f"Splitting dataset temporally | "
            f"train={train_ratio} | val={val_ratio} | test={test_ratio}"
        )

        # Sort by timestamp for temporal split
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        n  = len(df)

        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()

        logger.info(
            f"Split sizes → "
            f"train={len(train):,} | "
            f"val={len(val):,} | "
            f"test={len(test):,}"
        )

        return {"train": train, "val": val, "test": test}

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------

    def _save(self, splits: dict[str, pd.DataFrame]) -> None:
        """Save splits to Parquet format for efficient PyTorch loading."""
        for split_name, split_df in splits.items():
            path = self.output_dir / f"{split_name}.parquet"
            split_df.to_parquet(path, index=False, compression="snappy")
            logger.success(
                f"✅ Saved {split_name} → {path} "
                f"({len(split_df):,} rows, "
                f"{path.stat().st_size / 1024 / 1024:.1f} MB)"
            )

        # Save encoder mappings for inference
        encoders = {
            "n_users":      len(self.user_encoder.classes_),
            "n_items":      len(self.item_encoder.classes_),
            "n_categories": len(self.category_encoder.classes_),
            "user_classes": list(self.user_encoder.classes_),
            "item_classes": list(self.item_encoder.classes_),
            "cat_classes":  list(self.category_encoder.classes_),
        }
        import json
        encoder_path = self.output_dir / "encoders.json"
        with open(encoder_path, "w") as f:
            json.dump(encoders, f, indent=2)
        logger.success(f"✅ Encoder mappings saved → {encoder_path}")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    def _summarize(self, splits: dict[str, pd.DataFrame]) -> None:
        """Print a human-readable dataset summary."""
        train = splits["train"]
        logger.info("=" * 60)
        logger.info("  DATASET BUILD COMPLETE — SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total rows:       {sum(len(v) for v in splits.values()):>10,}")
        logger.info(f"  Train rows:       {len(splits['train']):>10,}")
        logger.info(f"  Val rows:         {len(splits['val']):>10,}")
        logger.info(f"  Test rows:        {len(splits['test']):>10,}")
        logger.info(f"  Unique users:     {train['user_id'].nunique():>10,}")
        logger.info(f"  Unique items:     {train['item_id'].nunique():>10,}")
        logger.info(f"  Feature columns:  {len(train.columns):>10,}")
        logger.info(f"  Positive rate:    {train['label_binary'].mean():>10.2%}")
        logger.info("=" * 60)


# ----------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time correct training dataset from Feast"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit interactions for development (default: use all)"
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default=str(DATA_RAW_DIR / "interactions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED_DIR),
    )
    return parser.parse_args()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    args = parse_args()

    builder = TrainingDatasetBuilder(
        interactions_path = Path(args.interactions_path),
        output_dir        = Path(args.output_dir),
        sample_size       = args.sample_size,
    )
    builder.build()


if __name__ == "__main__":
    main()