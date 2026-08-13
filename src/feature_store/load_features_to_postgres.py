"""
Feature Data Loader — CSV → PostgreSQL
=======================================
Loads the generated feature CSV files into PostgreSQL tables
so Feast's PostgreSQL offline store can read them natively.

This script runs ONCE after historical data generation (Step 2)
and then again after every Spark batch job (Step 6).

Pipeline position:
    [CSV files] ──► THIS SCRIPT ──► [PostgreSQL] ──► [Feast materialize] ──► [Redis]

Tables created:
    feast.user_features_raw   ← from data/raw/user_features.csv
    feast.item_features_raw   ← from data/raw/item_features.csv

Why a separate script and not direct Spark→PG write?
    In Step 6 (Airflow), Spark WILL write directly to PostgreSQL.
    This script is the manual bootstrap for development only.
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
# LOGGING SETUP
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
PG_PORT     = os.getenv("POSTGRES_PORT",     "5433")
PG_USER     = os.getenv("POSTGRES_USER",     "recsys_user")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "recsys_password")
PG_DB       = os.getenv("POSTGRES_DB",       "recsys_db")
PG_SCHEMA   = "feast"

DATA_RAW_DIR = Path(__file__).parent.parent / "data_generator" / "data" / "raw"

# Map: (csv filename) → (postgres table name)
FEATURE_TABLES = {
    "user_features.csv": "user_features_raw",
    "item_features.csv": "item_features_raw",
}


# ----------------------------------------------------------------
# LOADER CLASS
# ----------------------------------------------------------------

class FeaturePostgresLoader:
    """
    Loads feature CSVs into PostgreSQL for Feast offline store.

    Uses SQLAlchemy + pandas for reliable, typed loading.
    Idempotent: running it twice replaces data, never duplicates.
    """

    def __init__(self):
        self.engine = self._create_engine()
        self._ensure_schema_exists()

    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling."""
        conn_str = (
            f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
            f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
        )
        try:
            engine = create_engine(
                conn_str,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,     # Verify connection before use
                connect_args={
                    "connect_timeout": 10,
                    "options": f"-csearch_path={PG_SCHEMA}",
                }
            )
            # Test connection immediately
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.success(
                f"✅ Connected to PostgreSQL | "
                f"host={PG_HOST} | db={PG_DB}"
            )
            return engine
        except SQLAlchemyError as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            logger.error(
                "Is Docker running? Try: docker compose up -d postgres"
            )
            raise

    def _ensure_schema_exists(self) -> None:
        """Create the 'feast' schema if it doesn't exist."""
        with self.engine.connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA}")
            )
            conn.commit()
        logger.info(f"Schema '{PG_SCHEMA}' is ready")

    def load_table(
        self,
        csv_path:   Path,
        table_name: str,
    ) -> int:
        """
        Load a single CSV into a PostgreSQL table.

        Strategy: replace
            Drop and recreate the table on each load.
            This guarantees no stale data accumulates.
            In production with Spark, we'd use append + dedup.

        Args:
            csv_path:   Path to the CSV file
            table_name: Target PostgreSQL table name

        Returns:
            Number of rows loaded
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Run Step 2 data generator first."
            )

        logger.info(f"Loading: {csv_path.name} → {PG_SCHEMA}.{table_name}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"  Read {len(df):,} rows, {len(df.columns)} columns")

        # ---- Type enforcement ----
        # PostgreSQL needs explicit types for Feast to query correctly
        if "event_timestamp" in df.columns:
            df["event_timestamp"] = pd.to_datetime(
                df["event_timestamp"], utc=True
            )
        if "created_timestamp" in df.columns:
            df["created_timestamp"] = pd.to_datetime(
                df["created_timestamp"], utc=True
            )

        # Ensure numeric columns are proper types
        int_cols   = [c for c in df.columns if "_count" in c or "_idx" in c]
        float_cols = [c for c in df.columns if c in [
            "price", "avg_rating", "item_avg_rating_events",
            "item_cart_rate", "item_conversion_rate",
            "user_total_spend_30d", "user_avg_engagement_score",
        ]]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype("int64")
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype("float64")

        # ---- Write to PostgreSQL ----
        df.to_sql(
            name      = table_name,
            con       = self.engine,
            schema    = PG_SCHEMA,
            if_exists = "replace",   # Idempotent — safe to re-run
            index     = False,
            chunksize = 10_000,      # Batch inserts for large tables
            method    = "multi",     # Multi-row INSERT for speed
        )

        # ---- Create indexes for fast Feast queries ----
        self._create_indexes(table_name, df.columns.tolist())

        logger.success(
            f"✅ Loaded {len(df):,} rows → "
            f"{PG_SCHEMA}.{table_name}"
        )
        return len(df)

    def _create_indexes(self, table_name: str, columns: list[str]) -> None:
        """
        Create indexes on key columns for fast point-in-time queries.
        Feast queries filter heavily on entity keys and timestamps.
        """
        index_targets = []

        # Entity key indexes
        if "user_id" in columns:
            index_targets.append(("user_id", f"idx_{table_name}_user_id"))
        if "item_id" in columns:
            index_targets.append(("item_id", f"idx_{table_name}_item_id"))

        # Timestamp index — critical for point-in-time queries
        if "event_timestamp" in columns:
            index_targets.append((
                "event_timestamp",
                f"idx_{table_name}_event_ts"
            ))

        with self.engine.connect() as conn:
            for col, idx_name in index_targets:
                try:
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {PG_SCHEMA}.{table_name} ({col})"
                    ))
                    conn.commit()
                    logger.debug(f"  Index created: {idx_name} on ({col})")
                except Exception as e:
                    logger.warning(f"  Index creation skipped: {e}")

    def load_all(self) -> None:
        """Load all feature tables into PostgreSQL."""
        logger.info("=" * 60)
        logger.info("  FEATURE LOADING TO POSTGRESQL STARTED")
        logger.info("=" * 60)

        total_rows = 0
        start_time = datetime.utcnow()

        for csv_filename, table_name in FEATURE_TABLES.items():
            csv_path = DATA_RAW_DIR / csv_filename
            rows = self.load_table(csv_path, table_name)
            total_rows += rows

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("  LOADING COMPLETE")
        logger.info(f"  Total rows loaded: {total_rows:,}")
        logger.info(f"  Elapsed:           {elapsed:.1f}s")
        logger.info("=" * 60)

    def verify(self) -> None:
        """
        Query each table and print row counts.
        Quick sanity check after loading.
        """
        logger.info("Verifying loaded tables...")
        with self.engine.connect() as conn:
            for _, table_name in FEATURE_TABLES.items():
                try:
                    result = conn.execute(text(
                        f"SELECT COUNT(*) FROM {PG_SCHEMA}.{table_name}"
                    ))
                    count = result.scalar()
                    logger.success(
                        f"  ✅ {PG_SCHEMA}.{table_name}: {count:,} rows"
                    )
                except Exception as e:
                    logger.error(
                        f"  ❌ {PG_SCHEMA}.{table_name}: {e}"
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