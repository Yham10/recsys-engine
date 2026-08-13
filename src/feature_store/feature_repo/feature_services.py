"""
Feast Feature Services
======================
A FeatureService groups multiple FeatureViews into a single
named API that downstream consumers (training, serving) call.

Benefits:
    - Versioned: You can have v1, v2 of the same service
    - Decoupled: Consumers don't need to know which FeatureViews exist
    - Auditable: Track which features are used by which model

We define 2 services:
    ┌────────────────────────────────────────────────────────────┐
    │  training_feature_service  → All features for model train  │
    │  serving_feature_service   → Lean set for real-time serve  │
    └────────────────────────────────────────────────────────────┘
"""

from feast import FeatureService

from feature_views import (
    user_feature_view,
    item_feature_view,
)


# ----------------------------------------------------------------
# TRAINING FEATURE SERVICE
# Used by the Airflow training DAG to pull historical features.
# Includes ALL available features for maximum model expressiveness.
# ----------------------------------------------------------------
training_feature_service = FeatureService(
    name        = "training_feature_service",
    features    = [
        user_feature_view,   # All user features
        item_feature_view,   # All item features
    ],
    description = (
        "Complete feature set for offline model training. "
        "Retrieved via point-in-time joins to prevent data leakage."
    ),
    tags        = {
        "use_case":  "training",
        "model":     "two-tower-recommender",
        "version":   "v1",
    },
)


# ----------------------------------------------------------------
# SERVING FEATURE SERVICE
# Used by FastAPI at inference time.
# Only fetches USER features — item embeddings are pre-computed
# and cached separately (ANN index lookup, not feature store).
# This keeps inference latency minimal.
# ----------------------------------------------------------------
serving_feature_service = FeatureService(
    name        = "serving_feature_service",
    features    = [
        user_feature_view[
            [
                "user_click_count_7d",
                "user_purchase_count_30d",
                "user_total_spend_30d",
                "user_avg_engagement_score",
                "user_favorite_category",
            ]
        ],
    ],
    description = (
        "Minimal user feature set for real-time inference. "
        "Served from Redis with <10ms latency target."
    ),
    tags        = {
        "use_case":       "serving",
        "latency_target": "10ms",
        "store":          "redis",
        "version":        "v1",
    },
)