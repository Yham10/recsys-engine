"""
Feast Entity Definitions
========================
Entities are the primary keys of our feature store.
They define WHAT we are computing features FOR.

In our system:
    - user  → identified by user_id (string)
    - item  → identified by item_id (string)

Think of entities as the JOIN keys between your
feature tables and your training data.

Rule: Every FeatureView must be joined to an Entity.
"""

from feast import Entity, ValueType


# ----------------------------------------------------------------
# USER ENTITY
# ----------------------------------------------------------------
user_entity = Entity(
    name         = "user",
    join_keys    = ["user_id"],
    value_type   = ValueType.STRING,
    description  = (
        "A registered user of the e-commerce platform. "
        "Identified by a unique string user_id (e.g., 'user_000123')."
    ),
    tags         = {
        "team":        "recsys",
        "data_owner":  "ml-platform",
        "pii":         "true",     # Contains user data — flag for governance
    }
)


# ----------------------------------------------------------------
# ITEM ENTITY
# ----------------------------------------------------------------
item_entity = Entity(
    name         = "item",
    join_keys    = ["item_id"],
    value_type   = ValueType.STRING,
    description  = (
        "A product/item in the e-commerce catalog. "
        "Identified by a unique string item_id (e.g., 'item_000456')."
    ),
    tags         = {
        "team":       "recsys",
        "data_owner": "catalog-team",
        "pii":        "false",
    }
)