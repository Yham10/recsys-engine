"""
Recommendation Metrics
======================
"""

import torch
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field


@dataclass
class MetricResults:
    auc:         float = 0.0
    recall_k:    dict  = field(default_factory=dict)
    precision_k: dict  = field(default_factory=dict)
    ndcg_k:      dict  = field(default_factory=dict)
    mrr:         float = 0.0
    loss:        float = 0.0

    def to_dict(self, prefix: str = "") -> dict:
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
        parts = [f"AUC={self.auc:.4f}", f"Loss={self.loss:.4f}"]
        for k in sorted(self.recall_k):
            parts.append(f"Recall@{k}={self.recall_k[k]:.4f}")
        for k in sorted(self.ndcg_k):
            parts.append(f"NDCG@{k}={self.ndcg_k[k]:.4f}")
        parts.append(f"MRR={self.mrr:.4f}")
        return " | ".join(parts)


# ----------------------------------------------------------------
# METRIC FUNCTIONS
# ----------------------------------------------------------------

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.sum() == 0:
        return 0.0
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].sum() / labels.sum())


def compute_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].mean())


def compute_ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    top_k        = np.argsort(scores)[::-1][:k]
    top_k_labels = labels[top_k]
    discounts    = np.log2(np.arange(2, len(top_k_labels) + 2))
    dcg          = (top_k_labels / discounts).sum()

    ideal_labels = np.sort(labels)[::-1][:k]
    idcg         = (ideal_labels / discounts[:len(ideal_labels)]).sum()

    return float(dcg / idcg) if idcg > 1e-10 else 0.0


def compute_mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    for rank, idx in enumerate(np.argsort(scores)[::-1], start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


# ----------------------------------------------------------------
# EVALUATOR
# ----------------------------------------------------------------

class RecsysEvaluator:
    def __init__(self, k_values: list = None):
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        model:     torch.nn.Module,
        loader:    torch.utils.data.DataLoader,
        device:    torch.device,
        criterion: torch.nn.Module,
    ) -> MetricResults:
        model.eval()

        all_labels   = []
        all_scores   = []
        all_user_ids = []
        total_loss   = 0.0
        n_batches    = 0

        with torch.no_grad():
            for batch in loader:
                user_emb_idx  = batch["user_emb_idx"].to(device)
                item_emb_idx  = batch["item_emb_idx"].to(device)
                user_features = batch["user_features"].to(device)
                item_features = batch["item_features"].to(device)
                labels        = batch["label"].to(device)

                scores = model(
                    user_emb_idx, item_emb_idx,
                    user_features, item_features,
                )

                loss        = criterion(scores, labels)
                total_loss += loss.item()
                n_batches  += 1

                all_labels.append(labels.cpu().numpy())
                all_scores.append(torch.sigmoid(scores).cpu().numpy())

                # user_emb_idx[:, 0] is the 1-based user index
                # use it as a grouping key — identical for same user
                all_user_ids.append(user_emb_idx[:, 0].cpu().numpy())

        all_labels   = np.concatenate(all_labels)    # [N]
        all_scores   = np.concatenate(all_scores)    # [N]
        all_user_ids = np.concatenate(all_user_ids)  # [N]
        avg_loss     = total_loss / max(n_batches, 1)

        # ── AUC — global is correct for binary ranking ────────────
        auc = compute_auc(all_labels, all_scores)

        # ── FIX 3: Ranking metrics MUST be per-user then averaged ─
        # Computing over all 49K rows treats the entire val set as
        # one user's candidate list — that's not what these metrics mean
        recall_k_lists    = {k: [] for k in self.k_values}
        ndcg_k_lists      = {k: [] for k in self.k_values}
        precision_k_lists = {k: [] for k in self.k_values}
        mrr_list          = []

        n_users_evaluated = 0
        n_users_skipped   = 0

        for uid in np.unique(all_user_ids):
            mask     = (all_user_ids == uid)
            u_labels = all_labels[mask]
            u_scores = all_scores[mask]

            # Need at least 1 positive and 1 negative to compute ranking
            n_pos = u_labels.sum()
            n_neg = (1 - u_labels).sum()

            if n_pos == 0 or n_neg == 0 or len(u_labels) < 2:
                n_users_skipped += 1
                continue

            n_users_evaluated += 1
            mrr_list.append(compute_mrr(u_labels, u_scores))

            for k in self.k_values:
                recall_k_lists[k].append(
                    compute_recall_at_k(u_labels, u_scores, k)
                )
                ndcg_k_lists[k].append(
                    compute_ndcg_at_k(u_labels, u_scores, k)
                )
                precision_k_lists[k].append(
                    compute_precision_at_k(u_labels, u_scores, k)
                )

        if n_users_evaluated == 0:
            logger.warning(
                "No users had both positives and negatives in this split. "
                "Ranking metrics will be 0. Check your val/test split."
            )

        logger.debug(
            f"Evaluated {n_users_evaluated} users | "
            f"skipped {n_users_skipped} (no pos+neg pair)"
        )

        results = MetricResults(
            loss = avg_loss,
            auc  = auc,
            mrr  = float(np.mean(mrr_list)) if mrr_list else 0.0,
        )
        for k in self.k_values:
            results.recall_k[k]    = float(np.mean(recall_k_lists[k]))    if recall_k_lists[k]    else 0.0
            results.ndcg_k[k]      = float(np.mean(ndcg_k_lists[k]))      if ndcg_k_lists[k]      else 0.0
            results.precision_k[k] = float(np.mean(precision_k_lists[k])) if precision_k_lists[k] else 0.0

        model.train()
        return results