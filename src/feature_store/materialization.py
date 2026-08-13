"""
Feature Materialization Pipeline
=================================
Orchestrates the process of pushing batch-computed features
from the offline store (PostgreSQL/CSV) to the online store (Redis).

Called by:
    - Airflow DAG (Step 6): Scheduled daily after Spark batch job
    - Manual CLI: For backfills or emergency refreshes

Materialization Flow:
    CSV/PostgreSQL (offline) ──► feast materialize ──► Redis (online)
                                        ▲
                                   This script
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

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
logger.add(
    "logs/materialization.log",
    format="{time} | {level} | {name}:{line} | {message}",
    level="DEBUG",
    rotation="20 MB",
    retention="14 days",
)


# ----------------------------------------------------------------
# MATERIALIZATION RUNNER
# ----------------------------------------------------------------

class MaterializationRunner:
    """
    Manages feature materialization from offline to online store.

    Supports:
        - Full materialization (date range)
        - Incremental materialization (since last run)
        - Dry run (validate without writing)
    """

    def __init__(self):
        self.manager    = FeastManager()
        self.start_time = datetime.utcnow()

    def run_full(
        self,
        start_date: datetime,
        end_date:   datetime,
    ) -> None:
        """
        Materialize all features within a date range.
        Used for initial population or backfills.
        """
        logger.info("=" * 60)
        logger.info("  FULL MATERIALIZATION STARTED")
        logger.info(f"  Range: {start_date.date()} → {end_date.date()}")
        logger.info("=" * 60)

        self.manager.materialize_to_online_store(
            start_date = start_date,
            end_date   = end_date,
        )

        self._log_completion("FULL")

    def run_incremental(self) -> None:
        """
        Materialize only features newer than the last run.
        Default mode for scheduled daily runs.
        """
        logger.info("=" * 60)
        logger.info("  INCREMENTAL MATERIALIZATION STARTED")
        logger.info("=" * 60)

        self.manager.materialize_incremental(
            end_date=datetime.utcnow()
        )

        self._log_completion("INCREMENTAL")

    def run_dry_run(self) -> None:
        """
        Validate feature store state without writing anything.
        Useful for CI/CD pipeline validation.
        """
        logger.info("=" * 60)
        logger.info("  DRY RUN — Validating Feature Store")
        logger.info("=" * 60)

        # List all registered components
        self.manager.list_entities()
        self.manager.list_feature_views()

        # Print full stats
        stats = self.manager.get_feature_stats()
        logger.info("Feature Store Stats:")
        logger.info(f"  Project:          {stats['project']}")
        logger.info(f"  Feature Views:    {stats['n_feature_views']}")
        logger.info(f"  Entities:         {stats['n_entities']}")
        logger.info(f"  Feature Services: {stats['n_feature_services']}")

        for view in stats["feature_views"]:
            logger.info(
                f"\n  [{view['name']}]"
                f"\n    Features: {view['features']}"
                f"\n    TTL:      {view['ttl_days']} days"
                f"\n    Online:   {view['online']}"
            )

        logger.success("✅ Dry run complete — Feature Store is healthy")

    def _log_completion(self, mode: str) -> None:
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        logger.success(
            f"✅ {mode} materialization complete | "
            f"elapsed={elapsed:.1f}s"
        )


# ----------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature Store Materialization Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "dry-run"],
        default="incremental",
        help="Materialization mode"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for full mode (YYYY-MM-DD). Default: 7 days ago"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for full mode (YYYY-MM-DD). Default: today"
    )
    return parser.parse_args()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    args    = parse_args()
    runner  = MaterializationRunner()

    if args.mode == "full":
        end_date   = (
            datetime.fromisoformat(args.end_date)
            if args.end_date else datetime.utcnow()
        )
        start_date = (
            datetime.fromisoformat(args.start_date)
            if args.start_date else end_date - timedelta(days=7)
        )
        runner.run_full(start_date, end_date)

    elif args.mode == "incremental":
        runner.run_incremental()

    elif args.mode == "dry-run":
        runner.run_dry_run()


if __name__ == "__main__":
    main()