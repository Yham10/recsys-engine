"""
Serving Package
===============
Production FastAPI inference service for real-time recommendations.

SLA Targets:
    p50 latency: <50ms
    p99 latency: <100ms
    Throughput:  >500 req/sec (single instance)

Components:
    config.py           → Pydantic settings (env-driven)
    model_loader.py     → MLflow model + FAISS ANN index
    feature_fetcher.py  → Feast online store (Redis) integration
    recommender.py      → Core recommendation pipeline
    schemas.py          → API request/response contracts
    routers/            → FastAPI route handlers
    middleware.py       → Latency tracking, request tracing
    main.py             → App factory with lifespan management
"""