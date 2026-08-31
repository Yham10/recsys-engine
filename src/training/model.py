"""
Two-Tower Neural Network
========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_users:          int   = 10_000
    n_items:          int   = 5_000
    n_categories:     int   = 10

    user_embedding_dim:     int   = 64
    item_embedding_dim:     int   = 64
    category_embedding_dim: int   = 16

    n_user_continuous: int  = 4
    n_item_continuous: int  = 7

    user_hidden_layers: list = None
    item_hidden_layers: list = None

    output_dim:       int   = 64
    dropout_rate:     float = 0.2
    embedding_dropout: float = 0.1

    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-5
    batch_size:       int   = 2048

    def __post_init__(self):
        if self.user_hidden_layers is None:
            self.user_hidden_layers = [256, 128]
        if self.item_hidden_layers is None:
            self.item_hidden_layers = [256, 128]

    def to_dict(self) -> dict:
        return {
            "n_users":               self.n_users,
            "n_items":               self.n_items,
            "n_categories":          self.n_categories,
            "user_embedding_dim":    self.user_embedding_dim,
            "item_embedding_dim":    self.item_embedding_dim,
            "category_embedding_dim": self.category_embedding_dim,
            "n_user_continuous":     self.n_user_continuous,
            "n_item_continuous":     self.n_item_continuous,
            "user_hidden_layers":    self.user_hidden_layers,
            "item_hidden_layers":    self.item_hidden_layers,
            "output_dim":            self.output_dim,
            "dropout_rate":          self.dropout_rate,
            "embedding_dropout":     self.embedding_dropout,
            "learning_rate":         self.learning_rate,
            "weight_decay":          self.weight_decay,
            "batch_size":            self.batch_size,
        }

    def to_mlflow_params(self) -> dict:
        d = self.to_dict()
        d["user_hidden_layers"] = str(d["user_hidden_layers"])
        d["item_hidden_layers"] = str(d["item_hidden_layers"])
        return d


def build_mlp(
    input_dim:    int,
    hidden_dims:  list,
    output_dim:   int,
    dropout_rate: float = 0.2,
    use_batchnorm: bool = True,
) -> nn.Sequential:
    layers   = []
    in_dim   = input_dim
    all_dims = hidden_dims + [output_dim]

    for i, out_dim in enumerate(all_dims):
        layers.append(nn.Linear(in_dim, out_dim))

        if i < len(all_dims) - 1:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))

        in_dim = out_dim

    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # FIX 5: padding_idx=0 is reserved → indices must be 1-based
        # We allocate n_users+2 to be safe (+1 for shift, +1 for unknown)
        self.user_embedding = nn.Embedding(
            num_embeddings = config.n_users + 2,
            embedding_dim  = config.user_embedding_dim,
            padding_idx    = 0,
        )
        self.category_embedding = nn.Embedding(
            num_embeddings = config.n_categories + 2,
            embedding_dim  = config.category_embedding_dim,
            padding_idx    = 0,
        )
        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        emb_dim   = config.user_embedding_dim + config.category_embedding_dim
        input_dim = emb_dim + config.n_user_continuous

        self.mlp = build_mlp(
            input_dim    = input_dim,
            hidden_dims  = config.user_hidden_layers,
            output_dim   = config.output_dim,
            dropout_rate = config.dropout_rate,
        )

        self._init_weights()
        logger.debug(f"UserTower | input_dim={input_dim} | output_dim={config.output_dim}")

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.category_embedding.weight[0].fill_(0)

    def forward(
        self,
        user_emb_idx:  torch.Tensor,   # [B, 2]  values are 1-based
        user_features: torch.Tensor,   # [B, n_user_continuous]
    ) -> torch.Tensor:
        user_idx    = user_emb_idx[:, 0]
        fav_cat_idx = user_emb_idx[:, 1]

        user_emb = self.user_embedding(user_idx)
        cat_emb  = self.category_embedding(fav_cat_idx)

        user_emb = self.embedding_dropout(user_emb)
        cat_emb  = self.embedding_dropout(cat_emb)

        x = torch.cat([user_emb, cat_emb, user_features], dim=1)

        # FIX 2: Do NOT L2-normalize here — normalize only at the end
        # of TwoTowerModel.forward() after the dot product is scaled
        embedding = self.mlp(x)

        return embedding   # raw, un-normalized


class ItemTower(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.item_embedding = nn.Embedding(
            num_embeddings = config.n_items + 2,
            embedding_dim  = config.item_embedding_dim,
            padding_idx    = 0,
        )
        self.category_embedding = nn.Embedding(
            num_embeddings = config.n_categories + 2,
            embedding_dim  = config.category_embedding_dim,
            padding_idx    = 0,
        )
        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        emb_dim   = config.item_embedding_dim + config.category_embedding_dim
        input_dim = emb_dim + config.n_item_continuous

        self.mlp = build_mlp(
            input_dim    = input_dim,
            hidden_dims  = config.item_hidden_layers,
            output_dim   = config.output_dim,
            dropout_rate = config.dropout_rate,
        )

        self._init_weights()
        logger.debug(f"ItemTower | input_dim={input_dim} | output_dim={config.output_dim}")

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)
            self.category_embedding.weight[0].fill_(0)

    def forward(
        self,
        item_emb_idx:  torch.Tensor,   # [B, 2]  values are 1-based
        item_features: torch.Tensor,   # [B, n_item_continuous]
    ) -> torch.Tensor:
        item_idx = item_emb_idx[:, 0]
        cat_idx  = item_emb_idx[:, 1]

        item_emb = self.item_embedding(item_idx)
        cat_emb  = self.category_embedding(cat_idx)

        item_emb = self.embedding_dropout(item_emb)
        cat_emb  = self.embedding_dropout(cat_emb)

        x = torch.cat([item_emb, cat_emb, item_features], dim=1)

        embedding = self.mlp(x)

        return embedding   # raw, un-normalized


class TwoTowerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config     = config
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)

        # FIX 1: Remove learned temperature = 0.07
        # That value is for InfoNCE/contrastive loss, NOT for BCE
        # With L2-normalized embeddings, dot product ∈ [-1, 1]
        # We scale by a fixed constant to get logits in [-10, 10]
        # which gives sigmoid outputs spread across (0, 1)
        self.score_scale = 10.0

        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TwoTowerModel initialized | "
            f"total_params={total_params:,} | "
            f"trainable={trainable:,}"
        )

    def forward(
        self,
        user_emb_idx:  torch.Tensor,   # [B, 2]
        item_emb_idx:  torch.Tensor,   # [B, 2]
        user_features: torch.Tensor,   # [B, n_user_continuous]
        item_features: torch.Tensor,   # [B, n_item_continuous]
    ) -> torch.Tensor:
        user_emb = self.user_tower(user_emb_idx, user_features)
        item_emb = self.item_tower(item_emb_idx, item_features)

        # FIX 2: L2-normalize AFTER the MLP, right before dot product
        # This is the correct place — the towers output raw vectors,
        # normalization happens once here for the full model
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Cosine similarity ∈ [-1, 1]
        dot_product = (user_emb * item_emb).sum(dim=1)

        # FIX 1: Scale to reasonable logit range for BCE
        # sigmoid(-10) ≈ 0.00005, sigmoid(10) ≈ 0.99995
        # Gradients are non-zero across the full range
        scores = dot_product * self.score_scale

        return scores   # raw logits, BCEWithLogitsLoss applies sigmoid

    def get_user_embedding(
        self,
        user_emb_idx:  torch.Tensor,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized user embedding for ANN search."""
        with torch.no_grad():
            emb = self.user_tower(user_emb_idx, user_features)
            return F.normalize(emb, p=2, dim=1)

    def get_item_embeddings(
        self,
        item_emb_idx:  torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized item embeddings for ANN index."""
        with torch.no_grad():
            emb = self.item_tower(item_emb_idx, item_features)
            return F.normalize(emb, p=2, dim=1)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TwoTowerModel":
        return cls(config)