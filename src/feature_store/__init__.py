"""
Feature Store Package
=====================
Built on Feast (Feature Store for Machine Learning).

Responsibilities:
- Define and register all ML features (entities, feature views, services)
- Materialize batch features to offline store (PostgreSQL) for training
- Materialize real-time features to online store (Redis) for serving
- Generate point-in-time correct training datasets (no data leakage)
- Serve fresh features at inference time (<10ms from Redis)

Architecture:
    Offline Store (PostgreSQL) ──► Training Pipeline
    Online Store  (Redis)      ──► FastAPI Inference (<10ms)
"""