"""
Training Loop with MLflow Tracking
====================================
Manages the complete model training lifecycle:
    1. Load processed data via DataLoaders
    2. Initialize model, optimizer, scheduler
    3. Train with early stopping
    4. Track every metric, param, and artifact with MLflow
    5. Register the best model in MLflow Model Registry
    6. Save item embeddings for ANN index (used by FastAPI)

MLflow Tracking:
    Every run logs:
        Parameters  → all ModelConfig fields
        Metrics     → loss, AUC, Recall@K, NDCG@K per epoch
        Artifacts   → model weights, norm_stats.json, encoders.json
        Tags        → git commit, run type, dataset version
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from dataset import (
    build_dataloaders,
    USER_CONTINUOUS_COLS,
    ITEM_CONTINUOUS_COLS,
    NormalizationStats,
)
from model import TwoTowerModel, ModelConfig
from metrics import RecsysEvaluator, MetricResults


# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT_NAME", "recsys-recommendation-engine")
MLFLOW_MODEL_NAME    = os.getenv("MLFLOW_MODEL_NAME", "two-tower-recommender")

PROCESSED_DIR        = Path("data/processed")
ARTIFACTS_DIR        = Path("data/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# TRAINER
# ----------------------------------------------------------------

class TwoTowerTrainer:
    """
    Manages training, evaluation, and MLflow tracking
    for the Two-Tower recommendation model.

    Args:
        config:         ModelConfig with all hyperparameters
        processed_dir:  Path to train/val/test Parquet files
        n_epochs:       Maximum training epochs
        patience:       Early stopping patience (epochs)
        device:         'cuda', 'mps', or 'cpu'
    """

    def __init__(
        self,
        config:        ModelConfig,
        processed_dir: Path  = PROCESSED_DIR,
        n_epochs:      int   = 20,
        patience:      int   = 5,
        device:        str   = "auto",
        k_values:      list  = None,
    ):
        self.config        = config
        self.processed_dir = Path(processed_dir)
        self.n_epochs      = n_epochs
        self.patience      = patience
        self.k_values      = k_values or [5, 10, 20]

        # Device selection
        self.device = self._select_device(device)

        # Will be populated during train()
        self.model:     Optional[TwoTowerModel]  = None
        self.run_id:    Optional[str]            = None
        self.best_auc:  float                    = 0.0

        # Setup MLflow
        self._setup_mlflow()

        logger.info(
            f"TwoTowerTrainer initialized | "
            f"device={self.device} | "
            f"epochs={n_epochs} | "
            f"patience={patience}"
        )

    @staticmethod
    def _select_device(device: str) -> torch.device:
        """Auto-select best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                selected = "cuda"
            elif torch.backends.mps.is_available():
                selected = "mps"
            else:
                selected = "cpu"
        else:
            selected = device

        logger.info(f"Using device: {selected}")
        return torch.device(selected)

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Create experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            mlflow.create_experiment(
                name             = MLFLOW_EXPERIMENT,
                artifact_location = f"s3://mlflow-artifacts/{MLFLOW_EXPERIMENT}",
                tags             = {
                    "project":  "recsys-engine",
                    "team":     "ml-platform",
                }
            )
            logger.info(f"MLflow experiment created: {MLFLOW_EXPERIMENT}")

        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(
            f"MLflow configured | "
            f"uri={MLFLOW_TRACKING_URI} | "
            f"experiment={MLFLOW_EXPERIMENT}"
        )

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------

    def _load_encoder_config(self) -> None:
        """
        Load encoder mappings to set vocabulary sizes in ModelConfig.
        This ensures the embedding layers match the actual data.
        """
        encoder_path = self.processed_dir / "encoders.json"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"encoders.json not found at {encoder_path}.\n"
                f"Run Step 3 training_dataset.py first."
            )

        with open(encoder_path) as f:
            encoders = json.load(f)

        self.config.n_users      = encoders["n_users"]
        self.config.n_items      = encoders["n_items"]
        self.config.n_categories = encoders["n_categories"]

        logger.info(
            f"Encoder config loaded | "
            f"n_users={self.config.n_users:,} | "
            f"n_items={self.config.n_items:,} | "
            f"n_categories={self.config.n_categories}"
        )

    # ----------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------

    def train(self) -> str:
        """
        Full training pipeline.

        Returns:
            MLflow run_id of the completed run
        """
        # Load encoder config to set vocab sizes
        self._load_encoder_config()

        # Build DataLoaders
        train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
            processed_dir = self.processed_dir,
            batch_size    = self.config.batch_size,
            num_workers   = 0,      # Set 0 for Windows compatibility
            pin_memory    = (self.device.type == "cuda"),
        )

        # Initialize model
        self.model = TwoTowerModel(self.config).to(self.device)

        # Loss function — weighted BCE to handle class imbalance
        # Positive interactions (~30%) are weighted higher
        pos_weight = torch.tensor([2.5]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr           = self.config.learning_rate,
            weight_decay = self.config.weight_decay,
        )

        # LR Scheduler — reduce LR when val AUC plateaus
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode     = "max",      # Maximize AUC
            patience = 2,
            factor   = 0.5,
            min_lr   = 1e-6,
            # verbose  = True,
        )

        # Evaluator
        evaluator = RecsysEvaluator(k_values=self.k_values)

        # ---- Start MLflow Run ----
        with mlflow.start_run(run_name=f"two-tower-{datetime.now():%Y%m%d-%H%M}") as run:
            self.run_id = run.info.run_id
            logger.info(f"MLflow run started | run_id={self.run_id}")

            # Log all hyperparameters
            mlflow.log_params(self.config.to_dict())
            mlflow.log_params({
                "n_epochs":       self.n_epochs,
                "patience":       self.patience,
                "optimizer":      "Adam",
                "scheduler":      "ReduceLROnPlateau",
                "loss_fn":        "BCEWithLogitsLoss",
                "pos_weight":     2.5,
                "k_values":       str(self.k_values),
                "device":         str(self.device),
                "train_size":     len(train_loader.dataset),
                "val_size":       len(val_loader.dataset),
                "test_size":      len(test_loader.dataset),
            })

            # Log dataset artifact
            mlflow.log_artifact(
                str(self.processed_dir / "encoders.json"),
                artifact_path="dataset"
            )
            mlflow.log_artifact(
                str(self.processed_dir / "norm_stats.json"),
                artifact_path="dataset"
            )

            # Set tags
            mlflow.set_tags({
                "model_type":    "two-tower",
                "framework":     "pytorch",
                "data_version":  "v1",
                "run_type":      "training",
            })

            # ---- Epoch Loop ----
            best_val_auc    = 0.0
            epochs_no_improve = 0
            best_model_path = ARTIFACTS_DIR / "best_model.pt"

            for epoch in range(1, self.n_epochs + 1):
                epoch_start = time.time()

                # Training step
                train_loss = self._train_epoch(
                    train_loader, optimizer, criterion, epoch
                )

                # Validation step
                val_results = evaluator.evaluate(
                    self.model, val_loader, self.device, criterion
                )

                # LR scheduling based on val AUC
                scheduler.step(val_results.auc)
                current_lr = optimizer.param_groups[0]["lr"]

                epoch_time = time.time() - epoch_start

                # Log metrics to MLflow
                metrics = {
                    "train_loss":  train_loss,
                    "learning_rate": current_lr,
                    "epoch_time_s":  epoch_time,
                }
                metrics.update(val_results.to_dict(prefix="val_"))
                mlflow.log_metrics(metrics, step=epoch)

                logger.info(
                    f"Epoch {epoch:03d}/{self.n_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"{val_results} | "
                    f"lr={current_lr:.2e} | "
                    f"time={epoch_time:.1f}s"
                )

                # ---- Early Stopping & Model Checkpoint ----
                if val_results.auc > best_val_auc:
                    best_val_auc = val_results.auc
                    epochs_no_improve = 0

                    # Save best model checkpoint
                    torch.save({
                        "epoch":        epoch,
                        "model_state":  self.model.state_dict(),
                        "optimizer":    optimizer.state_dict(),
                        "val_auc":      best_val_auc,
                        "config":       self.config.to_dict(),
                    }, best_model_path)

                    logger.success(
                        f"  ✅ New best model saved | val_auc={best_val_auc:.4f}"
                    )
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"  No improvement for {epochs_no_improve}/{self.patience} epochs"
                    )

                    if epochs_no_improve >= self.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch}"
                        )
                        break

            # ---- Final Evaluation on Test Set ----
            logger.info("Loading best model for test evaluation...")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])

            test_results = evaluator.evaluate(
                self.model, test_loader, self.device, criterion
            )

            # Log test metrics
            mlflow.log_metrics(
                test_results.to_dict(prefix="test_"),
                step=self.n_epochs,
            )

            logger.info("=" * 60)
            logger.info("  FINAL TEST RESULTS")
            logger.info(f"  {test_results}")
            logger.info("=" * 60)

            # ---- Log Model to MLflow ----
            self._log_model_to_mlflow(
                norm_stats   = norm_stats,
                test_results = test_results,
            )

            # ---- Pre-compute Item Embeddings ----
            self._save_item_embeddings(
                test_loader  = test_loader,
            )

            logger.success(
                f"✅ Training complete | "
                f"run_id={self.run_id} | "
                f"best_val_auc={best_val_auc:.4f} | "
                f"test_auc={test_results.auc:.4f}"
            )

        return self.run_id

    def _train_epoch(
        self,
        loader:    torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch:     int,
    ) -> float:
        """
        Single training epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_idx, batch in enumerate(loader):
            # Move tensors to device
            user_emb_idx  = batch["user_emb_idx"].to(self.device)
            item_emb_idx  = batch["item_emb_idx"].to(self.device)
            user_features = batch["user_features"].to(self.device)
            item_features = batch["item_features"].to(self.device)
            labels        = batch["label"].to(self.device)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            scores = self.model(
                user_emb_idx,
                item_emb_idx,
                user_features,
                item_features,
            )

            # Compute loss
            loss = criterion(scores, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            # Log batch progress every 100 batches
            if batch_idx % 100 == 0:
                logger.debug(
                    f"  Epoch {epoch} | "
                    f"batch {batch_idx}/{len(loader)} | "
                    f"loss={loss.item():.4f}"
                )

        return total_loss / max(n_batches, 1)

    # ----------------------------------------------------------
    # MLFLOW MODEL LOGGING
    # ----------------------------------------------------------

    def _log_model_to_mlflow(
        self,
        norm_stats:   NormalizationStats,
        test_results: MetricResults,
    ) -> None:
        """
        Log the trained model + all artifacts to MLflow.
        Then register it in the Model Registry.
        """
        logger.info("Logging model to MLflow...")

        # Create a sample input for signature inference
        sample_input = {
            "user_emb_idx":  torch.zeros(1, 2, dtype=torch.long),
            "item_emb_idx":  torch.zeros(1, 2, dtype=torch.long),
            "user_features": torch.zeros(1, len(USER_CONTINUOUS_COLS)),
            "item_features": torch.zeros(1, len(ITEM_CONTINUOUS_COLS)),
        }

        self.model.eval()
        with torch.no_grad():
            sample_output = self.model(**sample_input)

        # Convert to numpy for signature inference
        sample_input_np  = {k: v.numpy() for k, v in sample_input.items()}
        sample_output_np = sample_output.numpy()

        signature = infer_signature(sample_input_np, sample_output_np)

        # Log model
        model_info = mlflow.pytorch.log_model(
            pytorch_model  = self.model,
            artifact_path  = "model",
            signature      = signature,
            registered_model_name = MLFLOW_MODEL_NAME,
            pip_requirements = [
                f"torch=={torch.__version__}",
                "numpy",
                "loguru",
            ],
        )

        # Log additional artifacts
        mlflow.log_artifact(
            str(self.processed_dir / "norm_stats.json"),
            artifact_path = "model"
        )
        mlflow.log_artifact(
            str(self.processed_dir / "encoders.json"),
            artifact_path = "model"
        )

        # Log model summary as text
        model_summary = (
            f"Two-Tower Model Summary\n"
            f"{'=' * 40}\n"
            f"Total params: {sum(p.numel() for p in self.model.parameters()):,}\n"
            f"Test AUC:     {test_results.auc:.4f}\n"
            f"Test MRR:     {test_results.mrr:.4f}\n"
        )
        for k in self.k_values:
            model_summary += (
                f"Test Recall@{k}: {test_results.recall_k[k]:.4f}\n"
                f"Test NDCG@{k}:   {test_results.ndcg_k[k]:.4f}\n"
            )

        with open(ARTIFACTS_DIR / "model_summary.txt", "w") as f:
            f.write(model_summary)
        mlflow.log_artifact(
            str(ARTIFACTS_DIR / "model_summary.txt"),
            artifact_path="model"
        )

        logger.success(
            f"✅ Model logged to MLflow | "
            f"model_uri={model_info.model_uri}"
        )

    # ----------------------------------------------------------
    # ITEM EMBEDDINGS (ANN INDEX INPUT)
    # ----------------------------------------------------------

    def _save_item_embeddings(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Pre-compute and save ALL item embeddings.

        These embeddings are used to build the ANN (Approximate
        Nearest Neighbor) index in FastAPI, so we can find the
        top-K most similar items to a user without scoring all items.

        Saved as:
            data/artifacts/item_embeddings.npy   [n_items, output_dim]
            data/artifacts/item_ids.json         [item_id list]
        """
        logger.info("Pre-computing item embeddings for ANN index...")

        self.model.eval()

        # Load encoders to get all item indices
        with open(self.processed_dir / "encoders.json") as f:
            encoders = json.load(f)

        n_items      = encoders["n_items"]
        item_classes = encoders["item_classes"]    # item_id strings

        # Build a batch of all item indices
        item_indices  = torch.arange(0, n_items, dtype=torch.long)
        cat_indices   = torch.zeros(n_items, dtype=torch.long)  # Default category
        item_emb_idx  = torch.stack([item_indices, cat_indices], dim=1)

        # Zero continuous features (embeddings capture catalog features)
        n_item_cont    = len(ITEM_CONTINUOUS_COLS)
        item_features  = torch.zeros(n_items, n_item_cont)

        # Process in batches to avoid OOM
        batch_size   = 512
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, n_items, batch_size):
                end = min(start + batch_size, n_items)
                emb = self.model.get_item_embeddings(
                    item_emb_idx[start:end].to(self.device),
                    item_features[start:end].to(self.device),
                )
                all_embeddings.append(emb.cpu().numpy())

        item_embeddings = np.concatenate(all_embeddings, axis=0)

        # Save
        emb_path = ARTIFACTS_DIR / "item_embeddings.npy"
        ids_path = ARTIFACTS_DIR / "item_ids.json"

        np.save(emb_path, item_embeddings)
        with open(ids_path, "w") as f:
            json.dump(item_classes, f)

        # Also log to MLflow
        mlflow.log_artifact(str(emb_path), artifact_path="embeddings")
        mlflow.log_artifact(str(ids_path), artifact_path="embeddings")

        logger.success(
            f"✅ Item embeddings saved | "
            f"shape={item_embeddings.shape} | "
            f"path={emb_path}"
        )