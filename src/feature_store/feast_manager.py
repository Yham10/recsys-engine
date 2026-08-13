"""
Feast Manager
=============
High-level wrapper around the Feast SDK.
All other components (training, serving, orchestration)
interact with the Feature Store through this class ONLY.

Design pattern: Facade
    - Hides Feast SDK complexity from consumers
    - Centralizes error handling and logging
    - Makes the feature store easily mockable in tests
    - Single place to swap out Feast for another store later

Usage:
    manager = FeastManager()

    # Training
    df = manager.get_training_data(entity_df)

    # Serving
    features = manager.get_online_features(user_id="user_000123")
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

from feast import FeatureStore
from feast.errors import FeatureViewNotFoundException


# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

FEATURE_REPO_PATH = Path(__file__).parent / "feature_repo"

# Features to retrieve for training
TRAINING_FEATURE_REFS = [
    # User features
    "user_features:user_click_count_7d",
    "user_features:user_purchase_count_30d",
    "user_features:user_total_spend_30d",
    "user_features:user_avg_engagement_score",
    "user_features:user_favorite_category",
    # Item features
    "item_features:item_view_count_7d",
    "item_features:item_purchase_count_30d",
    "item_features:item_avg_rating_events",
    "item_features:item_cart_rate",
    "item_features:item_conversion_rate",
    "item_features:price",
    "item_features:avg_rating",
    "item_features:category",
]

# Features to retrieve at serving time (user only — items pre-computed)
SERVING_USER_FEATURE_REFS = [
    "user_features:user_click_count_7d",
    "user_features:user_purchase_count_30d",
    "user_features:user_total_spend_30d",
    "user_features:user_avg_engagement_score",
    "user_features:user_favorite_category",
]


# ----------------------------------------------------------------
# FEAST MANAGER CLASS
# ----------------------------------------------------------------

class FeastManager:
    """
    Centralized interface for all Feature Store operations.

    Attributes:
        repo_path:  Path to the Feast feature repository
        store:      Initialized Feast FeatureStore instance
    """

    def __init__(self, repo_path: Path = FEATURE_REPO_PATH):
        self.repo_path = repo_path
        self.store     = self._initialize_store()

    def _initialize_store(self) -> FeatureStore:
        """
        Initialize the Feast FeatureStore.
        Validates the repo path exists and the store is reachable.
        """
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Feast feature repo not found at: {self.repo_path}\n"
                f"Ensure 'feature_store.yaml' exists in that directory."
            )
        try:
            store = FeatureStore(repo_path=str(self.repo_path))
            logger.info(
                f"✅ Feast FeatureStore initialized | "
                f"project={store.project} | "
                f"repo={self.repo_path}"
            )
            return store
        except Exception as e:
            logger.error(f"Failed to initialize Feast FeatureStore: {e}")
            raise

    # ==============================================================
    # OFFLINE OPERATIONS (Training)
    # ==============================================================

    def get_training_dataset(
        self,
        entity_df: pd.DataFrame,
        feature_refs: list[str] = TRAINING_FEATURE_REFS,
    ) -> pd.DataFrame:
        """
        Retrieve a point-in-time correct training dataset.

        This is the core of Train/Serve Skew prevention.
        Feast performs a point-in-time join:
            For each row in entity_df (user_id, item_id, timestamp),
            it fetches the feature values AS THEY WERE at that timestamp.

        This prevents the model from seeing "future" features —
        a common source of data leakage in naive implementations.

        Args:
            entity_df:    DataFrame with columns:
                          [user_id, item_id, event_timestamp, label]
            feature_refs: List of "feature_view:feature_name" strings

        Returns:
            DataFrame with entity columns + all requested features
        """
        logger.info(
            f"Retrieving training dataset | "
            f"rows={len(entity_df):,} | "
            f"features={len(feature_refs)}"
        )

        # Validate required columns
        required_cols = {"user_id", "item_id", "event_timestamp"}
        missing = required_cols - set(entity_df.columns)
        if missing:
            raise ValueError(
                f"entity_df is missing required columns: {missing}"
            )

        # Ensure timestamp column is proper datetime
        entity_df = entity_df.copy()
        entity_df["event_timestamp"] = pd.to_datetime(
            entity_df["event_timestamp"], utc=True
        )

        try:
            training_df = self.store.get_historical_features(
                entity_df    = entity_df,
                features     = feature_refs,
            ).to_df()

            logger.success(
                f"✅ Training dataset retrieved | "
                f"shape={training_df.shape} | "
                f"null_rate={training_df.isnull().mean().mean():.2%}"
            )
            return training_df

        except Exception as e:
            logger.error(f"Failed to retrieve training dataset: {e}")
            raise

    def build_entity_dataframe(
        self,
        interactions_path: str,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build the entity DataFrame from raw interaction data.

        The entity DataFrame tells Feast WHICH (user, item, timestamp)
        combinations to retrieve features for.

        Args:
            interactions_path:  Path to interactions.csv
            sample_size:        Optionally limit rows (for development)

        Returns:
            entity_df with columns: [user_id, item_id, event_timestamp, label]
        """
        logger.info(f"Building entity DataFrame from: {interactions_path}")

        df = pd.read_csv(interactions_path)

        # Use engagement_weight as training label
        entity_df = df[["user_id", "item_id", "timestamp", "engagement_weight"]].copy()
        entity_df = entity_df.rename(columns={
            "timestamp":         "event_timestamp",
            "engagement_weight": "label",
        })

        # Filter only positive interactions (label > 0) for training
        entity_df = entity_df[entity_df["label"] > 0].reset_index(drop=True)

        if sample_size:
            entity_df = entity_df.sample(
                n=min(sample_size, len(entity_df)),
                random_state=42
            ).reset_index(drop=True)
            logger.info(f"Sampled {len(entity_df):,} interactions")

        entity_df["event_timestamp"] = pd.to_datetime(
            entity_df["event_timestamp"], utc=True
        )

        logger.info(
            f"Entity DataFrame ready | "
            f"shape={entity_df.shape} | "
            f"unique_users={entity_df['user_id'].nunique():,} | "
            f"unique_items={entity_df['item_id'].nunique():,}"
        )
        return entity_df

    # ==============================================================
    # ONLINE OPERATIONS (Serving)
    # ==============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        reraise=True,
    )
    def get_online_user_features(self, user_id: str) -> dict:
        """
        Fetch real-time user features from Redis online store.

        This is called by FastAPI at inference time.
        Target latency: <10ms

        Args:
            user_id: The user to fetch features for

        Returns:
            Dict of feature_name -> feature_value

        Raises:
            KeyError:   If user_id not found in online store
            Exception:  If Redis is unreachable (after 3 retries)
        """
        logger.debug(f"Fetching online features | user_id={user_id}")

        try:
            response = self.store.get_online_features(
                features   = SERVING_USER_FEATURE_REFS,
                entity_rows = [{"user_id": user_id}],
            ).to_dict()

            # Flatten single-row response
            features = {
                k: v[0] for k, v in response.items()
                if k != "user_id"
            }

            # Check for None values (user not in online store)
            none_features = [k for k, v in features.items() if v is None]
            if none_features:
                logger.warning(
                    f"Missing online features for user={user_id}: "
                    f"{none_features}. Using default values."
                )
                features = self._apply_default_user_features(features)

            logger.debug(
                f"Online features fetched | "
                f"user_id={user_id} | features={list(features.keys())}"
            )
            return features

        except Exception as e:
            logger.error(
                f"Failed to fetch online features for user={user_id}: {e}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        reraise=True,
    )
    def get_online_item_features(self, item_ids: list[str]) -> pd.DataFrame:
        """
        Batch fetch real-time item features from Redis.

        Used when we need to re-rank candidate items at serving time.

        Args:
            item_ids: List of item IDs to fetch features for

        Returns:
            DataFrame indexed by item_id with feature columns
        """
        logger.debug(
            f"Fetching online item features | count={len(item_ids)}"
        )

        try:
            response = self.store.get_online_features(
                features    = [
                    "item_features:item_view_count_7d",
                    "item_features:item_purchase_count_30d",
                    "item_features:item_conversion_rate",
                    "item_features:item_cart_rate",
                    "item_features:price",
                    "item_features:avg_rating",
                    "item_features:category",
                ],
                entity_rows = [{"item_id": iid} for iid in item_ids],
            ).to_df()

            logger.debug(
                f"Item features fetched | shape={response.shape}"
            )
            return response

        except Exception as e:
            logger.error(f"Failed to fetch online item features: {e}")
            raise

    # ==============================================================
    # MATERIALIZATION
    # ==============================================================

    def materialize_to_online_store(
        self,
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> None:
        """
        Push features from the offline store to the online store (Redis).

        This is called by the Airflow DAG after each batch run.
        After materialization, the FastAPI service will serve
        the freshly computed features.

        Args:
            start_date: Materialize features from this date
            end_date:   Materialize features up to this date
                        (defaults to now)
        """
        end_date   = end_date   or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=2))

        logger.info(
            f"Materializing features to online store | "
            f"start={start_date.isoformat()} | "
            f"end={end_date.isoformat()}"
        )

        try:
            self.store.materialize(
                start_date = start_date,
                end_date   = end_date,
            )
            logger.success(
                "✅ Materialization complete — Redis updated with fresh features"
            )
        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            raise

    def materialize_incremental(
        self,
        end_date: Optional[datetime] = None
    ) -> None:
        """
        Incrementally push only NEW features since last materialization.
        More efficient for frequent updates (e.g., hourly runs).

        Args:
            end_date: Upper bound for materialization (defaults to now)
        """
        end_date = end_date or datetime.utcnow()

        logger.info(
            f"Running incremental materialization | end={end_date.isoformat()}"
        )

        try:
            self.store.materialize_incremental(end_date=end_date)
            logger.success("✅ Incremental materialization complete")
        except Exception as e:
            logger.error(f"Incremental materialization failed: {e}")
            raise

    # ==============================================================
    # REGISTRY OPERATIONS
    # ==============================================================

    def list_feature_views(self) -> list:
        """List all registered feature views."""
        views = self.store.list_feature_views()
        logger.info(f"Registered feature views ({len(views)}):")
        for view in views:
            logger.info(
                f"  → {view.name} | "
                f"entities={[e for e in view.entities]} | "
                f"features={len(view.features)} | "
                f"ttl={view.ttl}"
            )
        return views

    def list_entities(self) -> list:
        """List all registered entities."""
        entities = self.store.list_entities()
        logger.info(f"Registered entities ({len(entities)}):")
        for entity in entities:
            logger.info(f"  → {entity.name} | join_key={entity.join_key}")
        return entities

    def get_feature_stats(self) -> dict:
        """
        Return a summary dict of the feature store state.
        Used for monitoring and health checks.
        """
        views    = self.store.list_feature_views()
        entities = self.store.list_entities()
        services = self.store.list_feature_services()

        stats = {
            "project":          self.store.project,
            "n_feature_views":  len(views),
            "n_entities":       len(entities),
            "n_feature_services": len(services),
            "feature_views":    [
                {
                    "name":     v.name,
                    "features": [f.name for f in v.features],
                    "ttl_days": v.ttl.days if v.ttl else None,
                    "online":   v.online,
                }
                for v in views
            ],
        }
        return stats

    # ==============================================================
    # HELPERS
    # ==============================================================

    @staticmethod
    def _apply_default_user_features(features: dict) -> dict:
        """
        Apply sensible defaults for cold-start users
        (users not yet in the online store).

        Cold start is a real problem — a new user has no history.
        We use population averages as defaults instead of failing.
        """
        defaults = {
            "user_click_count_7d":      0,
            "user_purchase_count_30d":  0,
            "user_total_spend_30d":     0.0,
            "user_avg_engagement_score": 0.3,
            "user_favorite_category":   "electronics",  # Most popular category
        }
        return {
            k: (v if v is not None else defaults.get(k, 0))
            for k, v in features.items()
        }