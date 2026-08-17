"""
Recommendation Endpoints
=========================
Core API routes for generating recommendations.

POST /api/v1/recommend          → Main recommendation endpoint
GET  /api/v1/recommend/{user_id} → Convenience GET endpoint
GET  /api/v1/items/{item_id}    → Item detail with features
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from config import get_settings
from model_loader import get_registry
from recommender import RecommendationEngine
from feature_fetcher import FeatureFetcher
from schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    ErrorResponse,
)

router = APIRouter(
    prefix = "/api/v1",
    tags   = ["Recommendations"],
)


# ----------------------------------------------------------------
# DEPENDENCY: Recommendation Engine
# Built once per request from singletons initialized at startup
# ----------------------------------------------------------------

def get_engine(
    request:  Request,
    settings = Depends(get_settings),
) -> RecommendationEngine:
    """
    Dependency injection for the RecommendationEngine.
    Retrieves pre-initialized singletons from app state.
    """
    return request.app.state.engine


# ----------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------

@router.post(
    "/recommend",
    response_model  = RecommendationResponse,
    summary         = "Generate personalized recommendations",
    description     = (
        "Given a user_id, returns top-K personalized product recommendations "
        "using the Two-Tower neural network. Features are fetched in real-time "
        "from Redis (Feast online store). Target latency: <100ms."
    ),
    responses       = {
        200: {"description": "Recommendations generated successfully"},
        404: {"description": "User not found (cold start handled gracefully)"},
        503: {"description": "Model not loaded"},
    }
)
async def get_recommendations(
    body:     RecommendationRequest,
    request:  Request,
    engine:   RecommendationEngine = Depends(get_engine),
    settings  = Depends(get_settings),
) -> RecommendationResponse:
    """
    Main recommendation endpoint.

    Flow:
        1. Validate request (Pydantic, automatic)
        2. Check model is loaded
        3. Run recommendation pipeline
        4. Return formatted response with latency breakdown
    """
    registry   = get_registry()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Guard: model must be loaded
    if not registry.is_ready:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Service is starting up.",
        )

    try:
        recommendations, timings, is_cold_start = await engine.recommend(
            request    = body,
            request_id = request_id,
        )

        total_latency = sum(timings.values())

        return RecommendationResponse(
            user_id          = body.user_id,
            recommendations  = recommendations,
            top_k_requested  = body.top_k,
            top_k_returned   = len(recommendations),
            is_cold_start    = is_cold_start,
            latency_ms       = round(total_latency, 2),
            feature_latency_ms = round(timings.get("feature_ms", 0), 2),
            model_latency_ms   = round(timings.get("model_ms", 0), 2),
            ann_latency_ms     = round(timings.get("ann_ms", 0), 2),
            request_id       = request_id,
            served_at        = datetime.utcnow(),
            model_version    = registry.model_version,
        )

    except Exception as e:
        logger.error(
            f"Recommendation failed | "
            f"user={body.user_id} | "
            f"request_id={request_id} | "
            f"error={e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code = 500,
            detail      = f"Recommendation generation failed: {str(e)}",
        )


@router.get(
    "/recommend/{user_id}",
    response_model = RecommendationResponse,
    summary        = "Quick GET recommendation (convenience endpoint)",
)
async def get_recommendations_simple(
    user_id:  str,
    top_k:    int     = 10,
    request:  Request = None,
    engine:   RecommendationEngine = Depends(get_engine),
    settings  = Depends(get_settings),
) -> RecommendationResponse:
    """
    Convenience GET endpoint for quick testing.
    Same logic as POST /recommend but via URL params.
    """
    body = RecommendationRequest(user_id=user_id, top_k=top_k)
    return await get_recommendations(body, request, engine, settings)

@router.get(
    "/debug/{user_id}",
    summary     = "Debug endpoint — shows raw features for a user",
    description = "Shows what features Feast returns for a user. Dev only.",
)
async def debug_user(
    user_id:  str,
    request:  Request,
    engine:   RecommendationEngine = Depends(get_engine),
) -> dict:
    """
    Debug endpoint to inspect the full feature pipeline for a user.
    Shows exactly what the model sees at inference time.
    """
    registry = get_registry()

    # 1. Raw features from Feast
    raw_features, is_cold_start = engine.feature_fetcher.fetch_user_features(
        user_id=user_id
    )

    # 2. Tensor shapes after preprocessing
    user_emb_idx, user_continuous = engine.feature_fetcher.build_user_tensors(
        features = raw_features,
        user_id  = user_id,
        encoders = registry.encoders,
    )

    # 3. User embedding
    import torch
    with torch.no_grad():
        user_embedding = registry.model.get_user_embedding(
            user_emb_idx.to(registry.device),
            user_continuous.to(registry.device),
        )

    # 4. Sample metadata fetch
    sample_items = registry.item_ids[:3] if registry.item_ids else []
    metadata_sample = engine._fetch_item_metadata(sample_items)

    return {
        "user_id":              user_id,
        "is_cold_start":        is_cold_start,
        "raw_features":         raw_features,
        "user_emb_idx":         user_emb_idx.tolist(),
        "user_continuous_shape": list(user_continuous.shape),
        "user_embedding_shape": list(user_embedding.shape),
        "user_embedding_norm":  float(
            user_embedding.norm().item()
        ),
        "redis_has_features":   not is_cold_start,
        "db_metadata_working":  len(metadata_sample) > 0,
        "db_sample_result":     metadata_sample,
        "faiss_index_size":     registry.faiss_index.ntotal
                                if registry.faiss_index else 0,
        "model_version":        registry.model_version,
    }