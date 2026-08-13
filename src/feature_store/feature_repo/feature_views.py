"""
Feast Feature View Definitions
========================================
CHANGE from original:
    BEFORE: FileSource  → pointed at local CSV files
    AFTER:  PostgreSQLSource → points at PostgreSQL tables

This matches our feature_store.yaml which declares
offline_store: type: postgres.

Feast rule: offline_store type MUST match data source type.

    offline_store: postgres  ──► sources must be PostgreSQLSource ✅
    offline_store: file      ──► sources must be FileSource       ✅
    offline_store: postgres  ──► sources are FileSource           ❌ (our bug)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float64, Int64, String
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

from entities import user_entity, item_entity


# ----------------------------------------------------------------
# POSTGRESQL DATA SOURCES  (replaces FileSource)
# Points Feast at the tables loaded by load_features_to_postgres.py
# ----------------------------------------------------------------

PG_SCHEMA = "feast"

user_features_source = PostgreSQLSource(
    name             = "user_features_source",
    query            = f"SELECT * FROM {PG_SCHEMA}.user_features_raw",
    timestamp_field  = "event_timestamp",
    created_timestamp_column = "created_timestamp",
    description      = "Aggregated user behavioral features — PostgreSQL source",
    tags             = {"layer": "feature", "domain": "user"},
)

item_features_source = PostgreSQLSource(
    name             = "item_features_source",
    query            = f"SELECT * FROM {PG_SCHEMA}.item_features_raw",
    timestamp_field  = "event_timestamp",
    created_timestamp_column = "created_timestamp",
    description      = "Aggregated item engagement features — PostgreSQL source",
    tags             = {"layer": "feature", "domain": "item"},
)


# ================================================================
# FEATURE VIEWS
# Identical schema to before — only the source changed
# ================================================================

user_feature_view = FeatureView(
    name        = "user_features",
    entities    = [user_entity],
    ttl         = timedelta(days=2),
    schema      = [
        Field(
            name        = "user_click_count_7d",
            dtype       = Int64,
            description = "Number of item views in the last 7 days.",
        ),
        Field(
            name        = "user_purchase_count_30d",
            dtype       = Int64,
            description = "Number of completed purchases in last 30 days.",
        ),
        Field(
            name        = "user_total_spend_30d",
            dtype       = Float64,
            description = "Total amount spent (USD) in last 30 days.",
        ),
        Field(
            name        = "user_avg_engagement_score",
            dtype       = Float64,
            description = "Average engagement weight across all interactions.",
        ),
        Field(
            name        = "user_favorite_category",
            dtype       = String,
            description = "Most interacted product category.",
        ),
    ],
    source      = user_features_source,     # ← Now PostgreSQLSource
    online      = True,
    description = "User behavioral features — offline: PostgreSQL | online: Redis",
    tags        = {
        "team":          "recsys",
        "update_freq":   "daily",
        "serving_store": "redis",
    },
)


item_feature_view = FeatureView(
    name        = "item_features",
    entities    = [item_entity],
    ttl         = timedelta(days=2),
    schema      = [
        Field(
            name        = "item_view_count_7d",
            dtype       = Int64,
            description = "Number of views in last 7 days.",
        ),
        Field(
            name        = "item_purchase_count_30d",
            dtype       = Int64,
            description = "Number of purchases in last 30 days.",
        ),
        Field(
            name        = "item_avg_rating_events",
            dtype       = Float64,
            description = "Average rating from user rating events.",
        ),
        Field(
            name        = "item_cart_rate",
            dtype       = Float64,
            description = "Ratio of add-to-cart to item views.",
        ),
        Field(
            name        = "item_conversion_rate",
            dtype       = Float64,
            description = "Ratio of purchases to item views.",
        ),
        Field(
            name        = "price",
            dtype       = Float64,
            description = "Current item price in USD.",
        ),
        Field(
            name        = "avg_rating",
            dtype       = Float64,
            description = "Overall average rating from item catalog.",
        ),
        Field(
            name        = "category",
            dtype       = String,
            description = "Product category.",
        ),
    ],
    source      = item_features_source,     # ← Now PostgreSQLSource
    online      = True,
    description = "Item engagement features — offline: PostgreSQL | online: Redis",
    tags        = {
        "team":          "recsys",
        "update_freq":   "daily",
        "serving_store": "redis",
    },
)