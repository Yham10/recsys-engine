"""
Data Generator Entry Point
==========================
CLI tool for running historical data generation and/or
the real-time Kafka event stream.

Usage examples:
    # Generate all historical data:
    python run_generator.py --mode historical

    # Start real-time Kafka streaming:
    python run_generator.py --mode stream --rate 20

    # Generate data then immediately start streaming:
    python run_generator.py --mode both --rate 10 --max-events 5000
"""

import argparse
import sys
from loguru import logger
from pathlib import Path

# Configure loguru for production-style logging
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
    "logs/data_generator.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    level="DEBUG",
    rotation="50 MB",   # Rotate when file hits 50MB
    retention="7 days", # Keep logs for 7 days
    compression="zip",
)

from historical_data import HistoricalDataGenerator
from kafka_producer import ECommerceEventProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RecSys Data Generator — Historical & Real-Time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "stream", "both"],
        default="both",
        help="Operation mode"
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=10_000,
        help="Number of users to generate (historical mode)"
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=5_000,
        help="Number of items to generate (historical mode)"
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=500_000,
        help="Number of interactions to generate (historical mode)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Days of historical data to generate"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="Events per second for Kafka streaming"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop streaming after N events (default: run forever)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for generated data files"
    )
    return parser.parse_args()


def run_historical(args: argparse.Namespace) -> None:
    """Run the historical data generation pipeline."""
    logger.info("Starting HISTORICAL data generation mode...")

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    generator = HistoricalDataGenerator(
        n_users        = args.n_users,
        n_items        = args.n_items,
        n_interactions = args.n_interactions,
        days_back      = args.days_back,
        output_dir     = args.output_dir,
    )
    generator.generate_all()
    logger.success("✅ Historical data generation complete!")


def run_stream(args: argparse.Namespace) -> None:
    """Run the Kafka real-time event producer."""
    logger.info("Starting STREAM mode (Kafka event producer)...")

    producer = ECommerceEventProducer(
        events_per_second = args.rate,
    )
    producer.run(max_events=args.max_events)


def main() -> None:
    args = parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info(f"RecSys Data Generator starting | mode={args.mode}")

    if args.mode == "historical":
        run_historical(args)

    elif args.mode == "stream":
        run_stream(args)

    elif args.mode == "both":
        run_historical(args)
        logger.info("Historical data ready. Starting stream in 3 seconds...")
        import time; time.sleep(3)
        run_stream(args)


if __name__ == "__main__":
    main()