"""
Health & Readiness Endpoints
=============================
Critical for Kubernetes/Docker deployments.

GET /health  → Liveness probe  (is the app running?)
GET /ready   → Readiness probe (is the app ready to serve traffic?)
GET /metrics → Basic metrics   (request counts, latency stats)
"""

import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from config import get_settings
from model_loader import get_registry
from schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Infrastructure"])

# Track startup time for uptime calculation
_start_time = time.time()


@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness probe",
    description    = "Returns service health status. Used by Docker/K8s liveness probe.",
)
async def health_check(
    registry = Depends(get_registry),
    settings = Depends(get_settings),
) -> HealthResponse:
    """
    Liveness probe — checks if the service is alive.
    Returns 200 if healthy, 503 if degraded.
    """
    checks = {
        "model_loaded":    registry.model is not None,
        "faiss_loaded":    registry.faiss_index is not None,
        "encoders_loaded": registry.encoders is not None,
        "norm_stats_loaded": registry.norm_stats is not None,
    }

    # Redis check
    try:
        if hasattr(registry, "_feature_fetcher"):
            checks["redis_connected"] = registry._feature_fetcher.ping()
        else:
            checks["redis_connected"] = True   # Skip if not attached
    except Exception:
        checks["redis_connected"] = False

    all_healthy = all(checks.values())
    status = "healthy" if all_healthy else "degraded"

    response = HealthResponse(
        status           = status,
        version          = settings.app_version,
        environment      = settings.app_env,
        model_loaded     = checks.get("model_loaded", False),
        faiss_loaded     = checks.get("faiss_loaded", False),
        redis_connected  = checks.get("redis_connected", False),
        uptime_seconds   = time.time() - _start_time,
        checks           = checks,
    )

    if not all_healthy:
        raise HTTPException(status_code=503, detail=response.dict())

    return response


@router.get(
    "/ready",
    response_model = ReadinessResponse,
    summary        = "Readiness probe",
    description    = "Returns whether service is ready to serve traffic.",
)
async def readiness_check(
    registry = Depends(get_registry),
) -> ReadinessResponse:
    """
    Readiness probe — is the service ready to handle requests?
    Kubernetes won't route traffic until this returns 200.
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code = 503,
            detail      = ReadinessResponse(
                ready  = False,
                reason = "ML artifacts still loading"
            ).dict()
        )

    return ReadinessResponse(ready=True)