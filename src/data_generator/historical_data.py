"""
Historical Data Generator
=========================
Generates realistic static datasets that simulate months of
e-commerce activity. These files are used by:
  - Feast offline store (for point-in-time correct feature retrieval)
  - Model training pipeline 

Key insight: We model REAL user behavior patterns:
  - Users have category preferences (they don't buy randomly)
  - Popular items get disproportionately more interactions (power law)
  - User activity follows time-of-day patterns
  - Premium users interact more and spend more
"""

import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import Tuple

from schemas import (
    UserProfile, ItemProfile,
    EventType, ItemCategory, DeviceType
)


# ----------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Realistic probability distributions for event types
# Based on typical e-commerce funnel conversion rates
EVENT_TYPE_WEIGHTS = {
    EventType.ITEM_VIEW:        0.45,   # Most common action
    EventType.PAGE_VIEW:        0.25,
    EventType.SEARCH:           0.12,
    EventType.ADD_TO_CART:      0.08,
    EventType.WISHLIST_ADD:     0.04,
    EventType.PURCHASE:         0.03,   # ~3% conversion rate
    EventType.RATING:           0.02,
    EventType.REMOVE_FROM_CART: 0.01,
}

COUNTRIES = [
    "US", "UK", "DE", "FR", "CA",
    "AU", "JP", "BR", "IN", "MX"
]

BRANDS = {
    ItemCategory.ELECTRONICS:  ["Sony", "Samsung", "Apple", "LG", "Bosch"],
    ItemCategory.CLOTHING:     ["Nike", "Adidas", "Zara", "H&M", "Levis"],
    ItemCategory.BOOKS:        ["Penguin", "HarperCollins", "Random House"],
    ItemCategory.HOME_GARDEN:  ["IKEA", "Dyson", "Philips", "Cuisinart"],
    ItemCategory.SPORTS:       ["Nike", "Adidas", "Under Armour", "Puma"],
    ItemCategory.BEAUTY:       ["LOreal", "Maybelline", "Estee Lauder"],
    ItemCategory.TOYS:         ["LEGO", "Hasbro", "Mattel", "Fisher-Price"],
    ItemCategory.FOOD:         ["Nestle", "Kelloggs", "Organic Valley"],
    ItemCategory.AUTOMOTIVE:   ["Bosch", "3M", "Michelin", "Castrol"],
    ItemCategory.JEWELRY:      ["Pandora", "Swarovski", "Tiffany", "Zales"],
}

SUBCATEGORIES = {
    ItemCategory.ELECTRONICS:  ["Smartphones", "Laptops", "TVs", "Cameras", "Headphones"],
    ItemCategory.CLOTHING:     ["T-Shirts", "Jeans", "Dresses", "Shoes", "Jackets"],
    ItemCategory.BOOKS:        ["Fiction", "Non-Fiction", "Science", "History", "Tech"],
    ItemCategory.HOME_GARDEN:  ["Furniture", "Kitchen", "Garden", "Bedding", "Lighting"],
    ItemCategory.SPORTS:       ["Running", "Gym", "Swimming", "Cycling", "Yoga"],
    ItemCategory.BEAUTY:       ["Skincare", "Makeup", "Haircare", "Fragrance"],
    ItemCategory.TOYS:         ["Action Figures", "Board Games", "Puzzles", "Dolls"],
    ItemCategory.FOOD:         ["Snacks", "Beverages", "Organic", "Supplements"],
    ItemCategory.AUTOMOTIVE:   ["Car Care", "Tools", "Accessories", "Electronics"],
    ItemCategory.JEWELRY:      ["Rings", "Necklaces", "Earrings", "Bracelets"],
}

PRICE_RANGES = {
    ItemCategory.ELECTRONICS:  (50,   2000),
    ItemCategory.CLOTHING:     (10,   300),
    ItemCategory.BOOKS:        (5,    60),
    ItemCategory.HOME_GARDEN:  (15,   800),
    ItemCategory.SPORTS:       (10,   500),
    ItemCategory.BEAUTY:       (5,    200),
    ItemCategory.TOYS:         (8,    150),
    ItemCategory.FOOD:         (2,    80),
    ItemCategory.AUTOMOTIVE:   (10,   400),
    ItemCategory.JEWELRY:      (20,   5000),
}


# ----------------------------------------------------------------
# GENERATORS
# ----------------------------------------------------------------

class HistoricalDataGenerator:
    """
    Generates all static datasets for the ML pipeline.

    Attributes:
        n_users:        Number of unique users to simulate
        n_items:        Number of unique items in the catalog
        n_interactions: Total number of user-item interactions
        days_back:      How many days of history to generate
        output_dir:     Where to save the generated CSV files
    """

    def __init__(
        self,
        n_users: int        = 10_000,
        n_items: int        = 5_000,
        n_interactions: int = 500_000,
        days_back: int      = 90,
        output_dir: str     = "data/raw"
    ):
        self.n_users        = n_users
        self.n_items        = n_items
        self.n_interactions = n_interactions
        self.days_back      = days_back
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # These get populated by generate_users() and generate_items()
        self.users: list[UserProfile] = []
        self.items: list[ItemProfile] = []

        logger.info(
            f"HistoricalDataGenerator initialized | "
            f"users={n_users:,} | items={n_items:,} | "
            f"interactions={n_interactions:,} | days_back={days_back}"
        )

    # ----------------------------------------------------------
    # USER GENERATION
    # ----------------------------------------------------------

    def generate_users(self) -> pd.DataFrame:
        """
        Generate a realistic user base with demographic profiles.
        Each user has category preferences that will bias their
        interactions — making the recommendation problem non-trivial.
        """
        logger.info(f"Generating {self.n_users:,} user profiles...")

        users = []
        categories = [c.value for c in ItemCategory]

        for i in range(self.n_users):
            # Users prefer 1-3 categories (realistic behavior)
            n_preferred = random.randint(1, 3)
            preferred = random.sample(categories, n_preferred)

            user = {
                "user_id":                f"user_{i:06d}",
                "age":                    int(np.clip(np.random.normal(35, 12), 18, 80)),
                "gender":                 random.choices(
                                            ["M", "F", "Other"],
                                            weights=[0.48, 0.48, 0.04]
                                          )[0],
                "country":                random.choice(COUNTRIES),
                "preferred_categories":   ",".join(preferred),
                "account_age_days":       random.randint(1, 1825),  # Up to 5 years
                "is_premium":             random.random() < 0.15,   # 15% premium
            }
            users.append(user)

        self.users = users
        df = pd.DataFrame(users)
        output_path = self.output_dir / "users.csv"
        df.to_csv(output_path, index=False)
        logger.success(f"✅ Users saved → {output_path} ({len(df):,} rows)")
        return df

    # ----------------------------------------------------------
    # ITEM GENERATION
    # ----------------------------------------------------------

    def generate_items(self) -> pd.DataFrame:
        """
        Generate a product catalog with realistic attributes.
        Item popularity follows a power law distribution —
        a small % of items get the majority of interactions.
        """
        logger.info(f"Generating {self.n_items:,} item profiles...")

        items = []
        categories = list(ItemCategory)

        for i in range(self.n_items):
            category = random.choice(categories)
            cat_val  = category.value
            price_min, price_max = PRICE_RANGES[category]

            # Power law for review count — popular items dominate
            review_count = int(np.random.power(0.3) * 5000)

            item = {
                "item_id":      f"item_{i:06d}",
                "item_name":    f"{random.choice(BRANDS[category])} "
                                f"{random.choice(SUBCATEGORIES[category])} "
                                f"Model-{i:04d}",
                "category":     cat_val,
                "subcategory":  random.choice(SUBCATEGORIES[category]),
                "price":        round(random.uniform(price_min, price_max), 2),
                "avg_rating":   round(np.clip(np.random.normal(3.8, 0.7), 1, 5), 1),
                "review_count": review_count,
                "brand":        random.choice(BRANDS[category]),
                "is_available": random.random() < 0.92,  # 8% out of stock
            }
            items.append(item)

        self.items = items
        df = pd.DataFrame(items)
        output_path = self.output_dir / "items.csv"
        df.to_csv(output_path, index=False)
        logger.success(f"✅ Items saved → {output_path} ({len(df):,} rows)")
        return df

    # ----------------------------------------------------------
    # INTERACTION GENERATION
    # ----------------------------------------------------------

    def generate_interactions(self) -> pd.DataFrame:
        """
        Generate user-item interaction history.

        Key behaviors modeled:
        1. Users prefer items from their preferred categories (80% of the time)
        2. Popular items (high review_count) get more interactions (power law)
        3. Premium users have 2x more interactions and higher purchase rates
        4. Activity peaks during business hours and weekends
        5. Prices have ±10% variance (sales/promotions)
        """
        if not self.users or not self.items:
            raise RuntimeError(
                "Call generate_users() and generate_items() first."
            )

        logger.info(f"Generating {self.n_interactions:,} interactions...")

        users_df = pd.DataFrame(self.users)
        items_df = pd.DataFrame(self.items)

        # Build item popularity weights using review_count (power law)
        item_popularity = items_df["review_count"].values.astype(float)
        item_popularity = item_popularity / item_popularity.sum()

        interactions = []
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)

        # Group items by category for preference-based sampling
        items_by_category: dict[str, list] = {}
        for _, row in items_df.iterrows():
            cat = row["category"]
            if cat not in items_by_category:
                items_by_category[cat] = []
            items_by_category[cat].append(row["item_id"])

        event_types = list(EVENT_TYPE_WEIGHTS.keys())
        event_weights = list(EVENT_TYPE_WEIGHTS.values())

        for _ in range(self.n_interactions):
            # Sample a random user
            user = users_df.sample(1).iloc[0]
            user_id       = user["user_id"]
            is_premium    = user["is_premium"]
            preferred_cats = user["preferred_categories"].split(",")

            # 80% of the time, user interacts with preferred category
            if random.random() < 0.80 and preferred_cats:
                preferred_cat = random.choice(preferred_cats)
                candidate_items = items_by_category.get(preferred_cat, [])
                if candidate_items:
                    item_id = random.choice(candidate_items)
                else:
                    item_id = np.random.choice(
                        items_df["item_id"].values,
                        p=item_popularity
                    )
            else:
                # Explore outside preferences — popularity-weighted
                item_id = np.random.choice(
                    items_df["item_id"].values,
                    p=item_popularity
                )

            item_row = items_df[items_df["item_id"] == item_id].iloc[0]

            # Sample event type
            event_type = random.choices(event_types, weights=event_weights)[0]

            # Premium users purchase more
            if is_premium and event_type == EventType.ITEM_VIEW:
                if random.random() < 0.1:
                    event_type = EventType.PURCHASE

            # Generate a realistic timestamp
            # More activity during business hours (8am-10pm)
            random_seconds = random.randint(0, self.days_back * 86400)
            ts = start_date + timedelta(seconds=random_seconds)
            hour = ts.hour
            # Reject late-night timestamps 70% of the time (11pm-7am)
            if hour < 7 or hour > 23:
                if random.random() < 0.70:
                    ts = ts.replace(hour=random.randint(8, 22))

            # Price variance (±10% for sales)
            base_price    = item_row["price"]
            price_at_event = round(
                base_price * random.uniform(0.90, 1.10), 2
            )

            interaction = {
                "event_id":         str(uuid.uuid4()),
                "event_type":       event_type.value
                                    if isinstance(event_type, EventType)
                                    else event_type,
                "timestamp":        ts.isoformat(),
                "user_id":          user_id,
                "item_id":          item_id,
                "session_id":       str(uuid.uuid4()),
                "device_type":      random.choices(
                                      ["mobile", "desktop", "tablet"],
                                      weights=[0.55, 0.35, 0.10]
                                    )[0],
                "price_at_event":   price_at_event,
                "quantity":         random.randint(1, 3)
                                    if event_type in [
                                        EventType.PURCHASE, EventType.ADD_TO_CART
                                    ] else None,
                "rating_value":     round(random.uniform(1, 5), 1)
                                    if event_type == EventType.RATING else None,
                "engagement_weight": self._get_engagement_weight(event_type),
            }
            interactions.append(interaction)

        df = pd.DataFrame(interactions)

        # Sort by timestamp — critical for point-in-time feature retrieval
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        output_path = self.output_dir / "interactions.csv"
        df.to_csv(output_path, index=False)
        logger.success(
            f"✅ Interactions saved → {output_path} ({len(df):,} rows)"
        )
        return df

    # ----------------------------------------------------------
    # FEATURE SNAPSHOTS (for Feast offline store)
    # ----------------------------------------------------------

    def generate_user_features(
        self, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pre-aggregate user features from interaction history.
        These become the rows in our Feast UserFeatureView.

        Features computed:
            - user_click_count_7d:      Clicks in last 7 days
            - user_purchase_count_30d:  Purchases in last 30 days
            - user_total_spend_30d:     Total spend in last 30 days
            - user_avg_session_items:   Avg items per session
            - user_favorite_category:  Most interacted category
        """
        logger.info("Generating aggregated user features...")

        df = interactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max()

        # --- 7-day features ---
        df_7d = df[df["timestamp"] >= cutoff - timedelta(days=7)]
        clicks_7d = (
            df_7d[df_7d["event_type"] == "item_view"]
            .groupby("user_id")
            .size()
            .rename("user_click_count_7d")
        )

        # --- 30-day features ---
        df_30d = df[df["timestamp"] >= cutoff - timedelta(days=30)]
        purchases_30d = (
            df_30d[df_30d["event_type"] == "purchase"]
            .groupby("user_id")
            .size()
            .rename("user_purchase_count_30d")
        )
        spend_30d = (
            df_30d[df_30d["event_type"] == "purchase"]
            .groupby("user_id")["price_at_event"]
            .sum()
            .rename("user_total_spend_30d")
        )

        # --- All-time features ---
        avg_engagement = (
            df.groupby("user_id")["engagement_weight"]
            .mean()
            .rename("user_avg_engagement_score")
        )
        favorite_category = (
            df.merge(
                pd.read_csv(self.output_dir / "items.csv")[["item_id", "category"]],
                on="item_id",
                how="left"
            )
            .dropna(subset=["category"])
            .groupby("user_id")["category"]
            .agg(lambda x: x.value_counts().index[0])
            .rename("user_favorite_category")
        )

        # Combine all features
        users_df = pd.read_csv(self.output_dir / "users.csv")
        user_features = (
            users_df[["user_id"]]
            .merge(clicks_7d,       on="user_id", how="left")
            .merge(purchases_30d,   on="user_id", how="left")
            .merge(spend_30d,       on="user_id", how="left")
            .merge(avg_engagement,  on="user_id", how="left")
            .merge(favorite_category, on="user_id", how="left")
            .fillna(0)
        )

        # Feast requires an 'event_timestamp' column
        user_features["event_timestamp"] = cutoff
        user_features["created_timestamp"] = datetime.now()

        output_path = self.output_dir / "user_features.csv"
        user_features.to_csv(output_path, index=False)
        logger.success(
            f"✅ User features saved → {output_path} ({len(user_features):,} rows)"
        )
        return user_features

    def generate_item_features(
        self, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pre-aggregate item features from interaction history.
        These become rows in our Feast ItemFeatureView.

        Features computed:
            - item_view_count_7d:       Views in last 7 days
            - item_purchase_count_30d:  Purchases in last 30 days
            - item_avg_rating:          Average rating from events
            - item_cart_rate:           Add-to-cart / view ratio
            - item_conversion_rate:     Purchase / view ratio
        """
        logger.info("Generating aggregated item features...")

        df = interactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max()

        # 7-day views
        df_7d = df[df["timestamp"] >= cutoff - timedelta(days=7)]
        views_7d = (
            df_7d[df_7d["event_type"] == "item_view"]
            .groupby("item_id")
            .size()
            .rename("item_view_count_7d")
        )

        # 30-day purchases
        df_30d = df[df["timestamp"] >= cutoff - timedelta(days=30)]
        purchases_30d = (
            df_30d[df_30d["event_type"] == "purchase"]
            .groupby("item_id")
            .size()
            .rename("item_purchase_count_30d")
        )

        # Avg rating from events
        avg_rating = (
            df[df["event_type"] == "rating"]
            .groupby("item_id")["rating_value"]
            .mean()
            .rename("item_avg_rating_events")
        )

        # Conversion rates
        total_views = (
            df[df["event_type"] == "item_view"]
            .groupby("item_id")
            .size()
            .rename("total_views")
        )
        total_carts = (
            df[df["event_type"] == "add_to_cart"]
            .groupby("item_id")
            .size()
            .rename("total_carts")
        )
        total_purchases = (
            df[df["event_type"] == "purchase"]
            .groupby("item_id")
            .size()
            .rename("total_purchases")
        )

        # Combine
        items_df = pd.read_csv(self.output_dir / "items.csv")
        item_features = (
            items_df[["item_id", "price", "avg_rating", "category"]]
            .merge(views_7d,      on="item_id", how="left")
            .merge(purchases_30d, on="item_id", how="left")
            .merge(avg_rating,    on="item_id", how="left")
            .merge(total_views,   on="item_id", how="left")
            .merge(total_carts,   on="item_id", how="left")
            .merge(total_purchases, on="item_id", how="left")
            .fillna(0)
        )

        # Compute rates safely (avoid division by zero)
        item_features["item_cart_rate"] = np.where(
            item_features["total_views"] > 0,
            item_features["total_carts"] / item_features["total_views"],
            0.0
        ).round(4)

        item_features["item_conversion_rate"] = np.where(
            item_features["total_views"] > 0,
            item_features["total_purchases"] / item_features["total_views"],
            0.0
        ).round(4)

        # Drop intermediate columns
        item_features.drop(
            columns=["total_views", "total_carts", "total_purchases"],
            inplace=True
        )

        # Feast timestamp columns
        item_features["event_timestamp"]   = cutoff
        item_features["created_timestamp"] = datetime.now()

        output_path = self.output_dir / "item_features.csv"
        item_features.to_csv(output_path, index=False)
        logger.success(
            f"✅ Item features saved → {output_path} ({len(item_features):,} rows)"
        )
        return item_features

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    @staticmethod
    def _get_engagement_weight(event_type) -> float:
        weights = {
            "page_view":        0.1,
            "item_view":        0.3,
            "search":           0.2,
            "add_to_cart":      0.7,
            "remove_from_cart": -0.2,
            "purchase":         1.0,
            "rating":           0.8,
            "wishlist_add":     0.5,
        }
        key = event_type.value if isinstance(event_type, EventType) else event_type
        return weights.get(key, 0.1)

    def generate_all(self) -> Tuple[pd.DataFrame, ...]:
        """
        Master method — generates all datasets in the correct order.
        Returns all DataFrames for optional in-memory use.
        """
        logger.info("=" * 60)
        logger.info("  HISTORICAL DATA GENERATION STARTED")
        logger.info("=" * 60)

        users_df        = self.generate_users()
        items_df        = self.generate_items()
        interactions_df = self.generate_interactions()
        user_features   = self.generate_user_features(interactions_df)
        item_features   = self.generate_item_features(interactions_df)

        # Print summary statistics
        logger.info("=" * 60)
        logger.info("  GENERATION COMPLETE — SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Users:              {len(users_df):>10,}")
        logger.info(f"  Items:              {len(items_df):>10,}")
        logger.info(f"  Interactions:       {len(interactions_df):>10,}")
        logger.info(f"  User feature rows:  {len(user_features):>10,}")
        logger.info(f"  Item feature rows:  {len(item_features):>10,}")
        logger.info("=" * 60)

        return users_df, items_df, interactions_df, user_features, item_features