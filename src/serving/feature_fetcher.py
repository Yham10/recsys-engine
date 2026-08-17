"""
Feature Fetcher
===============
Fetches real-time user features from the Feast Online Store (Redis).
This is the bridge between FastAPI and the Feature Store.

At inference time:
    1. User makes request with user_id
    2. We fetch their LATEST features from Redis (<5ms)
    3. Features are preprocessed identically to training
    4. Features are fed into the user tower

Cold Start Handling:
    New users won't have features in Redis yet.
    We detect this and return population-average defaults.
    The recommendation won't be personalized but won't crash.
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from typing import Optional

from feast import FeatureStore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings
from dataset import (           # type: ignore
    USER_CONTINUOUS_COLS,
    USER_EMBEDDING_COLS,
    NormalizationStats,
)


# ----------------------------------------------------------------
# DEFAULT FEATURES FOR COLD START USERS
# Population averages — safe fallback when user has no history
# ----------------------------------------------------------------

COLD_START_DEFAULTS = {
    "user_click_count_7d":       0,
    "user_purchase_count_30d":   0,
    "user_total_spend_30d":      0.0,
    "user_avg_engagement_score": 0.3,    # Slight positive prior
    "user_favorite_category":    "electronics",
}

# Category name → integer index (must match training encoder)
CATEGORY_TO_IDX = {
    "electronics":  1,
    "clothing":     2,
    "books":        3,
    "home_garden":  4,
    "sports":       5,
    "beauty":       6,
    "toys":         7,
    "food":         8,
    "automotive":   9,
    "jewelry":      10,
}


class FeatureFetcher:
    """
    Real-time feature fetcher from Feast Online Store (Redis).

    Initialized once at app startup and reused across requests.
    Uses connection pooling internally via the Feast SDK.

    Args:
        feast_repo_path:  Path to Feast feature repository
        norm_stats:       NormalizationStats fitted during training
    """

    def __init__(
        self,
        feast_repo_path: str,
        norm_stats:      NormalizationStats,
    ):
        self.norm_stats = norm_stats
        self._store     = self._init_feast(feast_repo_path)
        logger.info("FeatureFetcher initialized")

    @staticmethod
    def _init_feast(repo_path: str) -> Optional[FeatureStore]:
        """
        Initialize Feast FeatureStore.
        Returns None if Feast is unavailable (graceful degradation).
        """
        try:
            store = FeatureStore(repo_path=repo_path)
            logger.success(
                f"✅ Feast connected | project={store.project}"
            )
            return store
        except Exception as e:
            logger.warning(
                f"Feast initialization failed: {e}\n"
                f"Will use cold-start defaults for all users."
            )
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.05, max=0.5),
        reraise=False,  # Don't raise — fall back to defaults
    )
    def fetch_user_features(
        self,
        user_id: str,
    ) -> tuple[dict, bool]:
        """
        Fetch user features from Feast Online Store (Redis).

        Args:
            user_id: User identifier

        Returns:
            (features_dict, is_cold_start)
            features_dict  → raw feature values
            is_cold_start  → True if user had no features in store
        """
        if self._store is None:
            logger.warning(
                f"Feast unavailable — using cold start for user={user_id}"
            )
            return COLD_START_DEFAULTS.copy(), True

        try:
            response = self._store.get_online_features(
                features=[
                    "user_features:user_click_count_7d",
                    "user_features:user_purchase_count_30d",
                    "user_features:user_total_spend_30d",
                    "user_features:user_avg_engagement_score",
                    "user_features:user_favorite_category",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            # Flatten single-row response
            features = {
                k: v[0] for k, v in response.items()
                if k != "user_id"
            }

            # Detect cold start (all values None)
            is_cold_start = all(v is None for v in features.values())

            if is_cold_start:
                logger.debug(
                    f"Cold start user detected: {user_id}"
                )
                return COLD_START_DEFAULTS.copy(), True

            # Replace any individual None values with defaults
            for key, default in COLD_START_DEFAULTS.items():
                if features.get(key) is None:
                    features[key] = default
                    logger.debug(
                        f"Feature '{key}' missing for user={user_id}, "
                        f"using default={default}"
                    )

            return features, False

        except Exception as e:
            logger.error(
                f"Feature fetch failed for user={user_id}: {e}. "
                f"Using cold start defaults."
            )
            return COLD_START_DEFAULTS.copy(), True

    def build_user_tensors(
        self,
        features:  dict,
        user_id:   str,
        encoders:  dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw feature dict into PyTorch tensors
        ready for the user tower forward pass.

        Applies identical preprocessing to training:
            1. Log-transform skewed features
            2. Z-score normalize using training stats
            3. Encode categorical → integer index
            4. Encode user_id → integer index (or 0 for unknown)

        Args:
            features:  Raw feature dict from Feast
            user_id:   String user ID
            encoders:  Encoder mappings from encoders.json

        Returns:
            (user_emb_idx, user_continuous)
                user_emb_idx:     [1, 2] LongTensor [user_idx, cat_idx]
                user_continuous:  [1, 4] FloatTensor normalized features
        """
        import numpy as np

        # ---- Encode user_id → integer index ----
        user_classes = encoders.get("user_classes", [])
        if user_id in user_classes:
            user_idx = user_classes.index(user_id)
        else:
            user_idx = 0    # Unknown user maps to padding index

        # ---- Encode favorite category → integer index ----
        fav_cat = features.get("user_favorite_category", "electronics")
        cat_idx = CATEGORY_TO_IDX.get(str(fav_cat).lower(), 0)

        # ---- Build continuous features (match training preprocessing) ----
        continuous = {
            # Log-transform (same as training_dataset.py)
            "user_click_count_7d_log": np.log1p(
                float(features.get("user_click_count_7d", 0))
            ),
            "user_purchase_count_30d": float(
                features.get("user_purchase_count_30d", 0)
            ),
            "user_total_spend_30d_log": np.log1p(
                float(features.get("user_total_spend_30d", 0.0))
            ),
            "user_avg_engagement_score": float(
                features.get("user_avg_engagement_score", 0.3)
            ),
        }

        # ---- Apply normalization (identical to training) ----
        means = self.norm_stats.means
        stds  = self.norm_stats.stds

        normalized = []
        for col in USER_CONTINUOUS_COLS:
            val  = continuous.get(col, 0.0)
            mean = means.get(col, 0.0)
            std  = stds.get(col, 1.0)
            normalized.append((val - mean) / (std + 1e-8))

        # ---- Build tensors ----
        user_emb_idx = torch.tensor(
            [[user_idx, cat_idx]], dtype=torch.long
        )                                               # [1, 2]

        user_continuous = torch.tensor(
            [normalized], dtype=torch.float32
        )                                               # [1, 4]

        return user_emb_idx, user_continuous

    def ping(self) -> bool:
        """Quick health check — can we reach the online store?"""
        if self._store is None:
            return False
        try:
            # Try a dummy feature fetch
            self._store.get_online_features(
                features   = ["user_features:user_click_count_7d"],
                entity_rows = [{"user_id": "__health_check__"}],
            )
            return True
        except Exception:
            return False