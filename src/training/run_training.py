"""
Training Entry Point
====================
CLI that launches the full training pipeline.
Called directly or by the Airflow DAG in Step 6.

Usage:
    # Default config
    python run_training.py

    # Custom hyperparameters
    python run_training.py \
        --epochs 30 \
        --lr 0.0005 \
        --batch-size 4096 \
        --embedding-dim 128 \
        --dropout 0.3
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

# ---- Logging Setup ----
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
    "logs/training.log",
    format="{time} | {level} | {name}:{line} | {message}",
    level="DEBUG",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
)

from model import ModelConfig
from trainer import TwoTowerTrainer, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Two-Tower Recommendation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model hyperparameters
    parser.add_argument("--embedding-dim", type=int,   default=64)
    parser.add_argument("--output-dim",    type=int,   default=64)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-5)
    parser.add_argument("--batch-size",    type=int,   default=2048)

    # Training config
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for Recall@K, NDCG@K metrics"
    )

    # Paths
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROCESSED_DIR),
    )

    return parser.parse_args()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  TWO-TOWER RECOMMENDER — TRAINING STARTED")
    logger.info("=" * 60)

    # Build model config from CLI args
    config = ModelConfig(
        user_embedding_dim     = args.embedding_dim,
        item_embedding_dim     = args.embedding_dim,
        category_embedding_dim = 16,
        output_dim             = args.output_dim,
        dropout_rate           = args.dropout,
        learning_rate          = args.lr,
        weight_decay           = args.weight_decay,
        batch_size             = args.batch_size,
    )

    # Launch trainer
    trainer = TwoTowerTrainer(
        config        = config,
        processed_dir = Path(args.processed_dir),
        n_epochs      = args.epochs,
        patience      = args.patience,
        device        = args.device,
        k_values      = args.k_values,
    )

    run_id = trainer.train()

    logger.success(f"✅ Training complete | MLflow run_id={run_id}")
    logger.info(
        f"View results: {trainer._setup_mlflow.__self__ if False else 'http://localhost:5000'}"
    )


if __name__ == "__main__":
    main()