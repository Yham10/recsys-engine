"""
Two-Tower Neural Network
========================
Industry-standard architecture for large-scale recommendation systems.
Used by: YouTube (2016), Google Play, Pinterest, Twitter.

Why Two Towers?
    A single model that takes (user, item) pairs doesn't scale —
    you'd need to run inference for every (user, item) combination
    at serving time. With 10K users × 5K items = 50M forward passes.

    Two towers separate the problem:
        USER TOWER:  user_id → user_embedding(64d)    [runs once per request]
        ITEM TOWER:  item_id → item_embedding(64d)    [pre-computed offline]

    At serving time:
        1. Compute user_embedding for the requesting user     (1 forward pass)
        2. Find top-K item_embeddings via ANN (FAISS/Redis)   (<5ms)
        3. No need to score all items at runtime              ✅

Training:
    Both towers train jointly via dot-product similarity + BCE loss.
    The gradient flows through BOTH towers simultaneously.

Inference:
    User tower → embedding → ANN search in pre-built item index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from dataclasses import dataclass


# ----------------------------------------------------------------
# MODEL CONFIG
# Centralized hyperparameters — logged to MLflow
# ----------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    All hyperparameters for the Two-Tower model.
    Passed to MLflow for experiment tracking.
    """
    # Vocabulary sizes (set from encoders.json)
    n_users:          int   = 10_000
    n_items:          int   = 5_000
    n_categories:     int   = 10

    # Embedding dimensions
    user_embedding_dim:     int   = 64
    item_embedding_dim:     int   = 64
    category_embedding_dim: int   = 16

    # Continuous feature dimensions
    n_user_continuous: int  = 4    # len(USER_CONTINUOUS_COLS)
    n_item_continuous: int  = 7    # len(ITEM_CONTINUOUS_COLS)

    # Tower hidden layer sizes
    user_hidden_layers: list = None
    item_hidden_layers: list = None

    # Output embedding dimension (must match for dot product)
    output_dim:       int   = 64

    # Regularization
    dropout_rate:     float = 0.2
    embedding_dropout: float = 0.1

    # Training
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-5
    batch_size:       int   = 2048

    def __post_init__(self):
        if self.user_hidden_layers is None:
            self.user_hidden_layers = [256, 128]
        if self.item_hidden_layers is None:
            self.item_hidden_layers = [256, 128]

    def to_dict(self) -> dict:
        """Serialize to dict for MLflow logging."""
        return {
            "n_users":               self.n_users,
            "n_items":               self.n_items,
            "n_categories":          self.n_categories,
            "user_embedding_dim":    self.user_embedding_dim,
            "item_embedding_dim":    self.item_embedding_dim,
            "category_embedding_dim": self.category_embedding_dim,
            "n_user_continuous":     self.n_user_continuous,
            "n_item_continuous":     self.n_item_continuous,
            "user_hidden_layers":    str(self.user_hidden_layers),
            "item_hidden_layers":    str(self.item_hidden_layers),
            "output_dim":            self.output_dim,
            "dropout_rate":          self.dropout_rate,
            "embedding_dropout":     self.embedding_dropout,
            "learning_rate":         self.learning_rate,
            "weight_decay":          self.weight_decay,
            "batch_size":            self.batch_size,
        }


# ----------------------------------------------------------------
# BUILDING BLOCKS
# ----------------------------------------------------------------

def build_mlp(
    input_dim:    int,
    hidden_dims:  list[int],
    output_dim:   int,
    dropout_rate: float = 0.2,
    use_batchnorm: bool = True,
) -> nn.Sequential:
    """
    Build a Multi-Layer Perceptron with:
        - BatchNorm (stabilizes training)
        - ReLU activation
        - Dropout (regularization)
        - L2 normalization on the final output embedding

    Args:
        input_dim:    Size of input tensor
        hidden_dims:  List of hidden layer sizes
        output_dim:   Final output embedding size
        dropout_rate: Dropout probability
        use_batchnorm: Whether to use BatchNorm layers

    Returns:
        nn.Sequential MLP
    """
    layers     = []
    in_dim     = input_dim
    all_dims   = hidden_dims + [output_dim]

    for i, out_dim in enumerate(all_dims):
        layers.append(nn.Linear(in_dim, out_dim))

        # No BN or activation on the final layer
        if i < len(all_dims) - 1:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))

        in_dim = out_dim

    return nn.Sequential(*layers)


# ----------------------------------------------------------------
# USER TOWER
# ----------------------------------------------------------------

class UserTower(nn.Module):
    """
    Encodes a user into a dense embedding vector.

    Input:
        user_emb_idx:  [batch, 2]  — [user_idx, user_fav_cat_idx]
        user_features: [batch, 4]  — continuous behavioral features

    Output:
        user_embedding: [batch, output_dim]  — L2-normalized
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings = config.n_users + 1,    # +1 for unknown users
            embedding_dim  = config.user_embedding_dim,
            padding_idx    = 0,
        )
        self.category_embedding = nn.Embedding(
            num_embeddings = config.n_categories + 1,
            embedding_dim  = config.category_embedding_dim,
            padding_idx    = 0,
        )
        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        # Compute MLP input dimension
        emb_dim   = config.user_embedding_dim + config.category_embedding_dim
        input_dim = emb_dim + config.n_user_continuous

        # MLP
        self.mlp = build_mlp(
            input_dim    = input_dim,
            hidden_dims  = config.user_hidden_layers,
            output_dim   = config.output_dim,
            dropout_rate = config.dropout_rate,
        )

        self._init_weights()
        logger.debug(
            f"UserTower initialized | "
            f"input_dim={input_dim} | "
            f"output_dim={config.output_dim}"
        )

    def _init_weights(self) -> None:
        """Xavier initialization for embedding layers."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        # Zero out padding embeddings
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.category_embedding.weight[0].fill_(0)

    def forward(
        self,
        user_emb_idx:  torch.Tensor,    # [B, 2]
        user_features: torch.Tensor,    # [B, n_user_continuous]
    ) -> torch.Tensor:
        """
        Returns:
            user_embedding: [B, output_dim]  L2-normalized
        """
        # Unpack embedding indices
        user_idx     = user_emb_idx[:, 0]   # [B]
        fav_cat_idx  = user_emb_idx[:, 1]   # [B]

        # Lookup embeddings
        user_emb    = self.user_embedding(user_idx)        # [B, user_emb_dim]
        cat_emb     = self.category_embedding(fav_cat_idx) # [B, cat_emb_dim]

        # Dropout on embeddings (regularize sparse features)
        user_emb = self.embedding_dropout(user_emb)
        cat_emb  = self.embedding_dropout(cat_emb)

        # Concatenate embeddings + continuous features
        x = torch.cat([user_emb, cat_emb, user_features], dim=1)

        # Pass through MLP
        embedding = self.mlp(x)

        # L2 normalize — makes dot product equivalent to cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


# ----------------------------------------------------------------
# ITEM TOWER
# ----------------------------------------------------------------

class ItemTower(nn.Module):
    """
    Encodes an item into a dense embedding vector.

    Input:
        item_emb_idx:  [batch, 2]  — [item_idx, item_cat_idx]
        item_features: [batch, 7]  — continuous engagement features

    Output:
        item_embedding: [batch, output_dim]  — L2-normalized
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.item_embedding = nn.Embedding(
            num_embeddings = config.n_items + 1,
            embedding_dim  = config.item_embedding_dim,
            padding_idx    = 0,
        )
        self.category_embedding = nn.Embedding(
            num_embeddings = config.n_categories + 1,
            embedding_dim  = config.category_embedding_dim,
            padding_idx    = 0,
        )
        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        # MLP input dimension
        emb_dim   = config.item_embedding_dim + config.category_embedding_dim
        input_dim = emb_dim + config.n_item_continuous

        # MLP
        self.mlp = build_mlp(
            input_dim    = input_dim,
            hidden_dims  = config.item_hidden_layers,
            output_dim   = config.output_dim,
            dropout_rate = config.dropout_rate,
        )

        self._init_weights()
        logger.debug(
            f"ItemTower initialized | "
            f"input_dim={input_dim} | "
            f"output_dim={config.output_dim}"
        )

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)
            self.category_embedding.weight[0].fill_(0)

    def forward(
        self,
        item_emb_idx:  torch.Tensor,    # [B, 2]
        item_features: torch.Tensor,    # [B, n_item_continuous]
    ) -> torch.Tensor:
        """
        Returns:
            item_embedding: [B, output_dim]  L2-normalized
        """
        item_idx = item_emb_idx[:, 0]   # [B]
        cat_idx  = item_emb_idx[:, 1]   # [B]

        item_emb = self.item_embedding(item_idx)
        cat_emb  = self.category_embedding(cat_idx)

        item_emb = self.embedding_dropout(item_emb)
        cat_emb  = self.embedding_dropout(cat_emb)

        x = torch.cat([item_emb, cat_emb, item_features], dim=1)

        embedding = self.mlp(x)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


# ----------------------------------------------------------------
# TWO-TOWER MODEL
# ----------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """
    Full Two-Tower recommendation model.

    Combines UserTower + ItemTower.
    Training objective: predict whether a user will interact with an item.
    Loss: Binary Cross-Entropy on dot-product similarity score.

    Forward pass returns a scalar score per (user, item) pair.
    At inference: only the user tower runs per-request.
                  item embeddings are pre-computed and indexed.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config      = config
        self.user_tower  = UserTower(config)
        self.item_tower  = ItemTower(config)

        # Learned temperature for scaling dot product
        # Helps control confidence of predictions
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            f"TwoTowerModel initialized | "
            f"total_params={total_params:,} | "
            f"trainable={trainable:,}"
        )

    def forward(
        self,
        user_emb_idx:  torch.Tensor,    # [B, 2]
        item_emb_idx:  torch.Tensor,    # [B, 2]
        user_features: torch.Tensor,    # [B, n_user_continuous]
        item_features: torch.Tensor,    # [B, n_item_continuous]
    ) -> torch.Tensor:
        """
        Returns:
            scores: [B]  scalar similarity score per pair
        """
        user_emb = self.user_tower(user_emb_idx, user_features)
        item_emb = self.item_tower(item_emb_idx, item_features)

        # Dot product similarity (both embeddings are L2-normalized)
        # Result is equivalent to cosine similarity ∈ [-1, 1]
        dot_product = (user_emb * item_emb).sum(dim=1)   # [B]

        # Scale by temperature (learned)
        scores = dot_product / self.temperature.clamp(min=1e-6)

        return scores

    def get_user_embedding(
        self,
        user_emb_idx:  torch.Tensor,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract user embedding only.
        Called by FastAPI at inference time.
        """
        with torch.no_grad():
            return self.user_tower(user_emb_idx, user_features)

    def get_item_embeddings(
        self,
        item_emb_idx:  torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract item embeddings for all items.
        Called once to pre-build the ANN index.
        """
        with torch.no_grad():
            return self.item_tower(item_emb_idx, item_features)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TwoTowerModel":
        return cls(config)