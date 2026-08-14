"""
Recommendation Metrics
======================
Standard IR (Information Retrieval) metrics for evaluating RecSys quality.

Metrics implemented:
    AUC-ROC     → Global ranking quality (pairwise)
    Recall@K    → Did we return the relevant item in top K? (coverage)
    Precision@K → Of top K returned, how many are relevant?
    NDCG@K      → Normalized Discounted Cumulative Gain (rank-aware quality)
    MRR         → Mean Reciprocal Rank (how high is the first relevant item?)

All metrics are computed at the batch level during training
and at the user level during full evaluation.
"""

import torch
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field


# ----------------------------------------------------------------
# METRIC RESULTS CONTAINER
# ----------------------------------------------------------------

@dataclass
class MetricResults:
    """Holds all evaluation metric values for a single evaluation run."""
    auc:        float = 0.0
    recall_k:   dict[int, float] = field(default_factory=dict)
    precision_k: dict[int, float] = field(default_factory=dict)
    ndcg_k:     dict[int, float] = field(default_factory=dict)
    mrr:        float = 0.0
    loss:       float = 0.0

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """
        Flatten to dict for MLflow logging.
        prefix = "val_" or "test_"
        """
        result = {
            f"{prefix}auc":  self.auc,
            f"{prefix}mrr":  self.mrr,
            f"{prefix}loss": self.loss,
        }
        for k, v in self.recall_k.items():
            result[f"{prefix}recall_at_{k}"] = v
        for k, v in self.precision_k.items():
            result[f"{prefix}precision_at_{k}"] = v
        for k, v in self.ndcg_k.items():
            result[f"{prefix}ndcg_at_{k}"] = v
        return result

    def __str__(self) -> str:
        parts = [
            f"AUC={self.auc:.4f}",
            f"Loss={self.loss:.4f}",
        ]
        for k in sorted(self.recall_k.keys()):
            parts.append(f"Recall@{k}={self.recall_k[k]:.4f}")
        for k in sorted(self.ndcg_k.keys()):
            parts.append(f"NDCG@{k}={self.ndcg_k[k]:.4f}")
        parts.append(f"MRR={self.mrr:.4f}")
        return " | ".join(parts)


# ----------------------------------------------------------------
# METRIC FUNCTIONS
# ----------------------------------------------------------------

def compute_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Compute AUC-ROC score.
    Handles edge case where only one class is present.

    Args:
        labels: Binary ground truth [N]
        scores: Predicted scores [N]

    Returns:
        AUC score in [0, 1]
    """
    if len(np.unique(labels)) < 2:
        logger.warning("AUC undefined — only one class in batch. Returning 0.5")
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_recall_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Recall@K: Fraction of relevant items that appear in top-K.

    For implicit feedback (binary labels):
        Recall@K = (# relevant items in top K) / (# total relevant items)

    Args:
        labels: Binary relevance [N]
        scores: Predicted scores [N]
        k:      Cutoff rank

    Returns:
        Recall@K in [0, 1]
    """
    if labels.sum() == 0:
        return 0.0

    top_k_indices = np.argsort(scores)[::-1][:k]
    n_relevant_in_k = labels[top_k_indices].sum()
    return float(n_relevant_in_k / labels.sum())


def compute_precision_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Precision@K: Fraction of top-K items that are relevant.

    Args:
        labels: Binary relevance [N]
        scores: Predicted scores [N]
        k:      Cutoff rank

    Returns:
        Precision@K in [0, 1]
    """
    top_k_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_indices].mean())


def compute_ndcg_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    NDCG penalizes relevant items found at lower ranks.
    A relevant item at rank 1 is worth more than at rank 10.

    NDCG@K = DCG@K / IDCG@K
    DCG@K  = Σ (rel_i / log2(i + 2))  for i in top K
    IDCG@K = DCG@K of the ideal (perfect) ranking

    Args:
        labels: Binary relevance [N]
        scores: Predicted scores [N]
        k:      Cutoff rank

    Returns:
        NDCG@K in [0, 1]
    """
    top_k_indices  = np.argsort(scores)[::-1][:k]
    top_k_labels   = labels[top_k_indices]

    # DCG
    discounts = np.log2(np.arange(2, len(top_k_labels) + 2))
    dcg = (top_k_labels / discounts).sum()

    # Ideal DCG (best possible ranking)
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = (ideal_labels / discounts[:len(ideal_labels)]).sum()

    if idcg < 1e-10:
        return 0.0
    return float(dcg / idcg)


def compute_mrr(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Mean Reciprocal Rank.
    1 / rank of the first relevant item in the ranked list.

    Args:
        labels: Binary relevance [N]
        scores: Predicted scores [N]

    Returns:
        MRR in [0, 1]
    """
    sorted_indices = np.argsort(scores)[::-1]
    for rank, idx in enumerate(sorted_indices, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


# ----------------------------------------------------------------
# EVALUATOR CLASS
# ----------------------------------------------------------------

class RecsysEvaluator:
    """
    Computes all recommendation metrics over a full dataset split.

    Usage:
        evaluator = RecsysEvaluator(k_values=[5, 10, 20])
        results = evaluator.evaluate(model, val_loader, device, criterion)
        mlflow.log_metrics(results.to_dict(prefix="val_"))
    """

    def __init__(self, k_values: list[int] = None):
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        model:     torch.nn.Module,
        loader:    torch.utils.data.DataLoader,
        device:    torch.device,
        criterion: torch.nn.Module,
    ) -> MetricResults:
        """
        Run full evaluation over a DataLoader.

        Args:
            model:     Trained TwoTowerModel
            loader:    DataLoader for val or test split
            device:    CPU or CUDA device
            criterion: Loss function

        Returns:
            MetricResults with all metrics populated
        """
        model.eval()

        all_labels = []
        all_scores = []
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in loader:
                # Move to device
                user_emb_idx  = batch["user_emb_idx"].to(device)
                item_emb_idx  = batch["item_emb_idx"].to(device)
                user_features = batch["user_features"].to(device)
                item_features = batch["item_features"].to(device)
                labels        = batch["label"].to(device)

                # Forward pass
                scores = model(
                    user_emb_idx,
                    item_emb_idx,
                    user_features,
                    item_features,
                )

                # Compute loss
                loss = criterion(scores, labels)
                total_loss += loss.item()
                n_batches  += 1

                # Collect predictions
                all_labels.append(labels.cpu().numpy())
                all_scores.append(
                    torch.sigmoid(scores).cpu().numpy()
                )

        # Concatenate all batches
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)

        avg_loss = total_loss / max(n_batches, 1)

        # Compute all metrics
        results = MetricResults(
            loss = avg_loss,
            auc  = compute_auc(all_labels, all_scores),
            mrr  = compute_mrr(all_labels, all_scores),
        )

        for k in self.k_values:
            results.recall_k[k]    = compute_recall_at_k(
                all_labels, all_scores, k
            )
            results.precision_k[k] = compute_precision_at_k(
                all_labels, all_scores, k
            )
            results.ndcg_k[k]      = compute_ndcg_at_k(
                all_labels, all_scores, k
            )

        model.train()
        return results