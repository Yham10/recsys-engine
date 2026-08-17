"""
FastAPI Application Factory
============================
Creates and configures the FastAPI application with:
    - Lifespan events (startup/shutdown)
    - Middleware (tracing, latency monitoring)
    - Routers (recommendations, health)
    - Exception handlers
    - OpenAPI documentation

Startup sequence:
    1. Load ML artifacts (model, FAISS, encoders, norm_stats)
    2. Initialize Feast connection (Redis)
    3. Build RecommendationEngine
    4. Mark service as ready

Shutdown sequence:
    1. Unload ML artifacts (release GPU memory)
    2. Close DB connections
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    "logs/serving.log",
    format="{time} | {level} | {name}:{line} | {message}",
    level="DEBUG",
    rotation="100 MB",
    retention="14 days",
    compression="zip",
)

from config import get_settings
from model_loader import load_all_artifacts, unload_artifacts, get_registry
from feature_fetcher import FeatureFetcher
from recommender import RecommendationEngine
from middleware import RequestTracingMiddleware, LatencyMonitoringMiddleware
from routers import health, recommendations


# ----------------------------------------------------------------
# LIFESPAN (startup + shutdown)
# ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the full application lifecycle.
    Code before yield → runs at startup.
    Code after yield  → runs at shutdown.
    """
    settings = get_settings()

    # ── STARTUP ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  Environment: {settings.app_env}")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    # 1. Load all ML artifacts into the global registry
    load_all_artifacts(settings)
    registry = get_registry()

    # 2. Initialize feature fetcher (Feast → Redis)
    feature_fetcher = FeatureFetcher(
        feast_repo_path = settings.feast_repo_path,
        norm_stats      = registry.norm_stats,
    )

    # 3. Build recommendation engine
    engine = RecommendationEngine(
        registry        = registry,
        feature_fetcher = feature_fetcher,
        settings        = settings,
    )

    # 4. Store singletons in app state for dependency injection
    app.state.engine          = engine
    app.state.feature_fetcher = feature_fetcher
    app.state.registry        = registry

    logger.success("✅ Service startup complete — ready to serve requests")

    yield  # Application runs here

    # ── SHUTDOWN ────────────────────────────────────────────────
    logger.info("Shutting down...")
    unload_artifacts()
    logger.info("✅ Shutdown complete")


# ----------------------------------------------------------------
# APP FACTORY
# ----------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    Using a factory function makes testing easier.
    """
    settings = get_settings()

    app = FastAPI(
        title           = settings.app_name,
        version         = settings.app_version,
        description     = (
            "Real-time e-commerce recommendation engine. "
            "Returns personalized product recommendations in <100ms "
            "using a Two-Tower neural network with Feast feature store."
        ),
        lifespan        = lifespan,
        docs_url        = "/docs",
        redoc_url       = "/redoc",
        openapi_url     = "/openapi.json",
    )

    # ── Middleware (order matters — outermost first) ─────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )
    app.add_middleware(LatencyMonitoringMiddleware, settings=settings)
    app.add_middleware(RequestTracingMiddleware)

    # ── Routers ──────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(recommendations.router)

    # ── Global Exception Handlers ────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Unhandled exception | "
            f"request_id={request_id} | "
            f"path={request.url.path} | "
            f"error={exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code = 500,
            content     = {
                "error":      "Internal server error",
                "request_id": request_id,
            }
        )

    return app


# ── Entry Point ──────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "main:app",
        host        = s.host,
        port        = s.port,
        reload      = s.debug,
        log_level   = s.log_level.lower(),
        access_log  = False,    # We handle logging in middleware
    )