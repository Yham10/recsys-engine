"""
Core Recommendation Engine
===========================
Orchestrates the full recommendation pipeline:

    1. Fetch user features from Redis       (~5ms)
    2. Run user tower → user embedding      (~2ms)
    3. FAISS ANN search → top-K item IDs   (~3ms)
    4. Apply filters (category, exclusions) (~0ms)
    5. Fetch item metadata from PostgreSQL  (~5ms)
    6. Format and return results            (~0ms)
                                           ──────
    Total:                                 ~15ms ✅
"""

import time
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from loguru import logger
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from config import get_settings
from feature_fetcher import FeatureFetcher
from model_loader import ModelRegistry
from schemas import RecommendedItem, RecommendationRequest


class RecommendationEngine:
    """
    Core engine that produces recommendations for a given user.

    Initialized once at startup and shared across all requests.
    All methods are stateless (no request-level state stored on self).

    Args:
        registry:        Loaded ModelRegistry (model + FAISS + encoders)
        feature_fetcher: FeatureFetcher connected to Redis
        settings:        Application settings
    """

    def __init__(
        self,
        registry:        ModelRegistry,
        feature_fetcher: FeatureFetcher,
        settings,
    ):
        self.registry        = registry
        self.feature_fetcher = feature_fetcher
        self.settings        = settings
        self._db_engine      = self._create_db_engine()

        logger.info("RecommendationEngine initialized")

    def _create_db_engine(self):
        """
        Create PostgreSQL connection pool for item metadata lookups.
        Uses connection pooling to avoid per-request connection overhead.
        """
        s = self.settings
        conn_str = (
            f"postgresql+psycopg2://{s.postgres_user}:{s.postgres_password}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
        )
        try:
            engine = create_engine(
                conn_str,
                poolclass    = QueuePool,
                pool_size    = 10,
                max_overflow = 20,
                pool_pre_ping = True,
                pool_recycle  = 300,
            )
            logger.info("PostgreSQL connection pool created")
            return engine
        except Exception as e:
            logger.warning(
                f"PostgreSQL unavailable: {e}. "
                f"Item metadata will not be included in responses."
            )
            return None

    # ----------------------------------------------------------
    # MAIN RECOMMENDATION METHOD
    # ----------------------------------------------------------

    async def recommend(
        self,
        request:    RecommendationRequest,
        request_id: str,
    ) -> tuple[list[RecommendedItem], dict]:
        """
        Generate top-K recommendations for a user.

        Args:
            request:    Validated recommendation request
            request_id: UUID for tracing

        Returns:
            (recommendations, latency_breakdown)
        """
        timings = {}
        user_id = request.user_id
        top_k   = request.top_k

        # ── Step 1: Fetch user features from Redis ──────────────
        t0 = time.perf_counter()

        raw_features, is_cold_start = self.feature_fetcher.fetch_user_features(
            user_id=user_id
        )
        user_emb_idx, user_continuous = self.feature_fetcher.build_user_tensors(
            features = raw_features,
            user_id  = user_id,
            encoders = self.registry.encoders,
        )

        timings["feature_ms"] = (time.perf_counter() - t0) * 1000

        # ── Step 2: User tower forward pass ─────────────────────
        t1 = time.perf_counter()

        user_emb_idx  = user_emb_idx.to(self.registry.device)
        user_continuous = user_continuous.to(self.registry.device)

        with torch.no_grad():
            user_embedding = self.registry.model.get_user_embedding(
                user_emb_idx  = user_emb_idx,
                user_features = user_continuous,
            )   # [1, output_dim]

        user_vec = user_embedding.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(user_vec)

        timings["model_ms"] = (time.perf_counter() - t1) * 1000

        # ── Step 3: ANN Search (FAISS) ───────────────────────────
        t2 = time.perf_counter()

        # Search for more candidates to allow for post-filtering
        n_candidates = min(top_k * 5, self.registry.faiss_index.ntotal)
        scores, indices = self.registry.faiss_index.search(
            user_vec, n_candidates
        )

        # scores:  [1, n_candidates]  similarity scores
        # indices: [1, n_candidates]  item integer indices

        scores  = scores[0]    # [n_candidates]
        indices = indices[0]   # [n_candidates]

        timings["ann_ms"] = (time.perf_counter() - t2) * 1000

        # ── Step 4: Apply Filters ────────────────────────────────
        # Build set of excluded item IDs
        excluded_ids = set(request.exclude_item_ids)

        # Filter invalid indices and excluded items
        valid_mask = indices >= 0
        filtered_scores  = scores[valid_mask]
        filtered_indices = indices[valid_mask]

        # Map integer indices → item_id strings
        candidate_items = []
        for idx, score in zip(filtered_indices, filtered_scores):
            if idx < len(self.registry.item_ids):
                item_id = self.registry.item_ids[idx]
                if item_id not in excluded_ids:
                    candidate_items.append((item_id, float(score)))

        # Score threshold filter
        candidate_items = [
            (iid, s) for iid, s in candidate_items
            if s >= self.settings.score_threshold
        ]

        # Limit to top_k
        candidate_items = candidate_items[:top_k * 2]

        # ── Step 5: Fetch Item Metadata ──────────────────────────
        item_metadata = self._fetch_item_metadata(
            item_ids = [iid for iid, _ in candidate_items]
        )

        # ── Step 6: Apply Category Filter & Format Results ───────
        recommendations = []
        rank = 1

        for item_id, score in candidate_items:
            if len(recommendations) >= top_k:
                break

            meta = item_metadata.get(item_id, {})

            # Category filter (applied here, not in FAISS)
            if request.category_filter:
                item_cat = meta.get("category", "")
                if item_cat.lower() != request.category_filter.lower():
                    continue

            recommendations.append(RecommendedItem(
                item_id     = item_id,
                rank        = rank,
                score       = round(score, 6),
                item_name   = meta.get("item_name"),
                category    = meta.get("category"),
                price       = meta.get("price"),
                avg_rating  = meta.get("avg_rating"),
                explanation = self._generate_explanation(
                    score, is_cold_start, meta.get("category"),
                    raw_features.get("user_favorite_category")
                ),
            ))
            rank += 1

        logger.info(
            f"Recommendations generated | "
            f"user={user_id} | "
            f"returned={len(recommendations)}/{top_k} | "
            f"cold_start={is_cold_start} | "
            f"feature={timings['feature_ms']:.1f}ms | "
            f"model={timings['model_ms']:.1f}ms | "
            f"ann={timings['ann_ms']:.1f}ms"
        )

        return recommendations, timings, is_cold_start

    # ----------------------------------------------------------
    # ITEM METADATA
    # ----------------------------------------------------------

    def _fetch_item_metadata(
        self,
        item_ids: list[str],
    ) -> dict[str, dict]:
        """
        Batch fetch item metadata from PostgreSQL.
        Queries the enriched item_features_raw table which
        contains both feature columns AND catalog columns
        (item_name, brand, is_available).
        """
        if not item_ids:
            logger.warning("_fetch_item_metadata called with empty item_ids")
            return {}

        if self._db_engine is None:
            logger.warning(
                "DB engine is None — metadata fetch skipped. "
                "Check PostgreSQL connection settings."
            )
            return {}

        try:
            placeholders = ", ".join(
                [f":id_{i}" for i in range(len(item_ids))]
            )
            params = {
                f"id_{i}": iid for i, iid in enumerate(item_ids)
            }

            # Query only columns we KNOW exist in the enriched table
            query = text(f"""
                SELECT
                    item_id,
                    category,
                    price,
                    avg_rating,
                    item_name,
                    brand,
                    is_available,
                    item_view_count_7d,
                    item_purchase_count_30d,
                    item_conversion_rate
                FROM feast.item_features_raw
                WHERE item_id IN ({placeholders})
            """)

            with self._db_engine.connect() as conn:
                result = conn.execute(query, params)
                rows   = result.fetchall()
                cols   = list(result.keys())

            if not rows:
                logger.warning(
                    f"No metadata rows returned for {len(item_ids)} items. "
                    f"Sample: {item_ids[:3]}. "
                    f"Table may be empty — run load_features_to_postgres.py"
                )
                return {}

            metadata = {
                row[0]: dict(zip(cols, row))
                for row in rows
            }

            logger.debug(
                f"Metadata fetched | "
                f"requested={len(item_ids)} | "
                f"returned={len(metadata)}"
            )
            return metadata

        except Exception as e:
            logger.error(
                f"_fetch_item_metadata FAILED: {e}",
                exc_info=True,
            )
            return {}

    # ----------------------------------------------------------
    # EXPLANATION GENERATION
    # ----------------------------------------------------------

    @staticmethod
    def _generate_explanation(
        score:          float,
        is_cold_start:  bool,
        item_category:  Optional[str],
        user_fav_cat:   Optional[str],
    ) -> str:
        """
        Generate a human-readable explanation for the recommendation.
        Useful for debugging and user-facing "Why recommended?" features.
        """
        if is_cold_start:
            return "Trending in your region"

        if item_category and user_fav_cat:
            if str(item_category).lower() == str(user_fav_cat).lower():
                return f"Matches your interest in {item_category}"

        if score > 0.8:
            return "Highly relevant to your browsing history"
        elif score > 0.5:
            return "Based on your recent activity"
        else:
            return "You might also like"