"""
Feature Data Loader — CSV → PostgreSQL  (FIXED)
=======================================
Changes from v1:
    - Merges items.csv + item_features.csv before loading
      so item_name, brand, is_available are available for
      the FastAPI metadata lookup
    - Merges users.csv + user_features.csv for completeness
    - Added explicit column validation before loading
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ----------------------------------------------------------------
# LOGGING
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

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
PG_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
PG_PORT     = os.getenv("POSTGRES_PORT",     "5432")
PG_USER     = os.getenv("POSTGRES_USER",     "recsys_user")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "recsys_password")
PG_DB       = os.getenv("POSTGRES_DB",       "recsys_db")
PG_SCHEMA   = "feast"

# ---- Resolve data directory ----
# Adjust this path to wherever your CSVs actually live
DATA_RAW_DIR = (
    Path(__file__).parent.parent
    / "data_generator"
    / "data"
    / "raw"
)


# ----------------------------------------------------------------
# LOADER
# ----------------------------------------------------------------

class FeaturePostgresLoader:
    """
    Loads feature CSVs into PostgreSQL for Feast offline store.
    Merges catalog data (items.csv, users.csv) with computed
    feature aggregations before loading.
    """

    def __init__(self):
        self.engine = self._create_engine()
        self._ensure_schema_exists()

    def _create_engine(self):
        conn_str = (
            f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
            f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
        )
        try:
            engine = create_engine(
                conn_str,
                pool_size     = 5,
                max_overflow  = 10,
                pool_pre_ping = True,
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.success(
                f"✅ Connected to PostgreSQL | "
                f"host={PG_HOST} | db={PG_DB}"
            )
            return engine
        except SQLAlchemyError as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    def _ensure_schema_exists(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA}")
            )
            conn.commit()
        logger.info(f"Schema '{PG_SCHEMA}' is ready")

    # ----------------------------------------------------------
    # BUILD ENRICHED DATAFRAMES
    # ----------------------------------------------------------

    def _build_item_table(self) -> pd.DataFrame:
        """
        Merge items.csv (catalog) + item_features.csv (aggregations).

        items.csv has:          item_name, brand, is_available, price, ...
        item_features.csv has:  item_view_count_7d, item_conversion_rate, ...

        Result table has ALL columns needed by:
            - Feast feature views (feature columns)
            - FastAPI metadata lookup (item_name, brand, etc.)
        """
        items_path    = DATA_RAW_DIR / "items.csv"
        features_path = DATA_RAW_DIR / "item_features.csv"

        # Validate files exist
        for p in [items_path, features_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    f"Run Step 2 data generator first."
                )

        logger.info(f"Loading items catalog from:    {items_path}")
        logger.info(f"Loading item features from:    {features_path}")

        items_df    = pd.read_csv(items_path)
        features_df = pd.read_csv(features_path)

        logger.info(
            f"items.csv columns:         {list(items_df.columns)}"
        )
        logger.info(
            f"item_features.csv columns: {list(features_df.columns)}"
        )

        # Drop duplicate columns that exist in both
        # (price, avg_rating, category already in item_features.csv)
        catalog_only_cols = [
            "item_id", "item_name", "subcategory",
            "brand", "is_available", "review_count"
        ]
        # Keep only catalog-specific columns + item_id for the join
        catalog_cols = [
            c for c in catalog_only_cols
            if c in items_df.columns
        ]
        items_subset = items_df[catalog_cols]

        # Merge on item_id
        merged = features_df.merge(
            items_subset,
            on  = "item_id",
            how = "left",
        )

        # ---- Type enforcement ----
        if "event_timestamp" in merged.columns:
            merged["event_timestamp"] = pd.to_datetime(
                merged["event_timestamp"], utc=True
            )
        if "created_timestamp" in merged.columns:
            merged["created_timestamp"] = pd.to_datetime(
                merged["created_timestamp"], utc=True
            )

        int_cols = [
            "item_view_count_7d",
            "item_purchase_count_30d",
            "review_count",
        ]
        float_cols = [
            "price", "avg_rating",
            "item_avg_rating_events",
            "item_cart_rate",
            "item_conversion_rate",
        ]
        bool_cols = ["is_available"]

        for col in int_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype("int64")
        for col in float_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0).astype("float64")
        for col in bool_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(True).astype(bool)

        logger.info(
            f"Enriched item table ready | "
            f"rows={len(merged):,} | "
            f"columns={list(merged.columns)}"
        )
        return merged

    def _build_user_table(self) -> pd.DataFrame:
        """
        Merge users.csv (profile) + user_features.csv (aggregations).
        """
        users_path    = DATA_RAW_DIR / "users.csv"
        features_path = DATA_RAW_DIR / "user_features.csv"

        for p in [users_path, features_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    f"Run Step 2 data generator first."
                )

        logger.info(f"Loading users catalog from:    {users_path}")
        logger.info(f"Loading user features from:    {features_path}")

        users_df    = pd.read_csv(users_path)
        features_df = pd.read_csv(features_path)

        # Keep only profile columns not already in features
        profile_cols = [
            c for c in ["user_id", "age", "gender", "country",
                        "account_age_days", "is_premium"]
            if c in users_df.columns
        ]
        users_subset = users_df[profile_cols]

        merged = features_df.merge(
            users_subset,
            on  = "user_id",
            how = "left",
        )

        # Type enforcement
        if "event_timestamp" in merged.columns:
            merged["event_timestamp"] = pd.to_datetime(
                merged["event_timestamp"], utc=True
            )
        if "created_timestamp" in merged.columns:
            merged["created_timestamp"] = pd.to_datetime(
                merged["created_timestamp"], utc=True
            )

        int_cols = [
            "user_click_count_7d",
            "user_purchase_count_30d",
            "account_age_days",
            "age",
        ]
        float_cols = [
            "user_total_spend_30d",
            "user_avg_engagement_score",
        ]

        for col in int_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype("int64")
        for col in float_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0).astype("float64")

        logger.info(
            f"Enriched user table ready | "
            f"rows={len(merged):,} | "
            f"columns={list(merged.columns)}"
        )
        return merged

    # ----------------------------------------------------------
    # LOAD TO POSTGRESQL
    # ----------------------------------------------------------

    def _load_dataframe(
        self,
        df:         pd.DataFrame,
        table_name: str,
    ) -> int:
        """
        Load a DataFrame into a PostgreSQL table.
        Idempotent — replaces data on each run.
        """
        logger.info(
            f"Loading {len(df):,} rows → "
            f"{PG_SCHEMA}.{table_name} ..."
        )

        df.to_sql(
            name      = table_name,
            con       = self.engine,
            schema    = PG_SCHEMA,
            if_exists = "replace",
            index     = False,
            chunksize = 5_000,
            method    = "multi",
        )

        self._create_indexes(table_name, df.columns.tolist())

        logger.success(
            f"✅ {len(df):,} rows loaded → "
            f"{PG_SCHEMA}.{table_name}"
        )
        return len(df)

    def _create_indexes(
        self,
        table_name: str,
        columns:    list[str],
    ) -> None:
        """Create indexes on entity keys and timestamp columns."""
        targets = []

        if "user_id" in columns:
            targets.append(("user_id", f"idx_{table_name}_user_id"))
        if "item_id" in columns:
            targets.append(("item_id", f"idx_{table_name}_item_id"))
        if "event_timestamp" in columns:
            targets.append(
                ("event_timestamp", f"idx_{table_name}_event_ts")
            )

        with self.engine.connect() as conn:
            for col, idx_name in targets:
                try:
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {PG_SCHEMA}.{table_name} ({col})"
                    ))
                    conn.commit()
                    logger.debug(f"Index ready: {idx_name}")
                except Exception as e:
                    logger.warning(f"Index skipped: {idx_name} — {e}")

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def load_all(self) -> None:
        """Load all enriched feature tables into PostgreSQL."""
        logger.info("=" * 60)
        logger.info("  FEATURE LOADING TO POSTGRESQL STARTED")
        logger.info(f"  Source dir: {DATA_RAW_DIR}")
        logger.info("=" * 60)

        start = datetime.utcnow()

        # Build enriched DataFrames
        item_df = self._build_item_table()
        user_df = self._build_user_table()

        # Load to PostgreSQL
        item_rows = self._load_dataframe(item_df, "item_features_raw")
        user_rows = self._load_dataframe(user_df, "user_features_raw")

        elapsed = (datetime.utcnow() - start).total_seconds()

        logger.info("=" * 60)
        logger.info("  LOADING COMPLETE")
        logger.info(f"  item_features_raw: {item_rows:,} rows")
        logger.info(f"  user_features_raw: {user_rows:,} rows")
        logger.info(f"  Elapsed:           {elapsed:.1f}s")
        logger.info("=" * 60)

    def verify(self) -> None:
        """Print row counts and column list for each table."""
        logger.info("Verifying loaded tables...")
        tables = ["item_features_raw", "user_features_raw"]

        with self.engine.connect() as conn:
            for table in tables:
                try:
                    count = conn.execute(
                        text(
                            f"SELECT COUNT(*) "
                            f"FROM {PG_SCHEMA}.{table}"
                        )
                    ).scalar()

                    cols = conn.execute(
                        text(
                            f"SELECT column_name "
                            f"FROM information_schema.columns "
                            f"WHERE table_schema = '{PG_SCHEMA}' "
                            f"AND table_name = '{table}' "
                            f"ORDER BY ordinal_position"
                        )
                    ).fetchall()

                    col_names = [c[0] for c in cols]

                    logger.success(
                        f"  ✅ {PG_SCHEMA}.{table}: "
                        f"{count:,} rows | "
                        f"columns={col_names}"
                    )
                except Exception as e:
                    logger.error(
                        f"  ❌ {PG_SCHEMA}.{table}: {e}"
                    )


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    loader = FeaturePostgresLoader()
    loader.load_all()
    loader.verify()


if __name__ == "__main__":
    main()