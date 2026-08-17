"""
Application Configuration
=========================
All settings are loaded from environment variables.
Pydantic-settings validates types and provides defaults.

Never hardcode secrets — read from .env or Docker secrets.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All fields have sensible defaults for local development.
    """

    # ---- Application ----
    app_name:       str  = "RecSys Inference API"
    app_version:    str  = "0.1.0"
    app_env:        str  = Field(default="development", env="APP_ENV")
    log_level:      str  = Field(default="INFO",        env="LOG_LEVEL")
    debug:          bool = Field(default=False,          env="DEBUG")

    # ---- Server ----
    host:           str  = "0.0.0.0"
    port:           int  = 8000
    workers:        int  = 1       # Use 1 for model-loaded services

    # ---- MLflow ----
    mlflow_tracking_uri:  str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_model_name:    str = Field(
        default="two-tower-recommender",
        env="MLFLOW_MODEL_NAME"
    )
    mlflow_model_stage:   str = Field(
        default="latest",
        env="MLFLOW_MODEL_STAGE"
    )

    # ---- Redis (Feast Online Store) ----
    redis_host:     str  = Field(default="localhost", env="REDIS_HOST")
    redis_port:     int  = Field(default=6379,         env="REDIS_PORT")
    redis_db:       int  = Field(default=0,            env="REDIS_DB")
    redis_password: str  = Field(default="",           env="REDIS_PASSWORD")

    # ---- PostgreSQL (Item Metadata) ----
    postgres_host:     str = Field(default="localhost",      env="POSTGRES_HOST")
    postgres_port:     int = Field(default=5432,             env="POSTGRES_PORT")
    postgres_user:     str = Field(default="recsys_user",    env="POSTGRES_USER")
    postgres_password: str = Field(default="recsys_password",env="POSTGRES_PASSWORD")
    postgres_db:       str = Field(default="recsys_db",      env="POSTGRES_DB")

    # ---- Feast ----
    feast_repo_path: str = Field(
        default="src/feature_store/feature_repo",
        env="FEAST_REPO_PATH"
    )

    # ---- Artifacts ----
    artifacts_dir: Path = Field(
        default=Path("src/training/data/artifacts"),
        env="ARTIFACTS_DIR"
    )
    processed_dir: Path = Field(
        default=Path("src/training/data/processed"),
        env="PROCESSED_DIR"
    )

    # ---- Recommendation Config ----
    default_top_k:        int   = 10
    max_top_k:            int   = 50
    min_top_k:            int   = 1
    faiss_n_probe:        int   = 10    # ANN search quality vs speed tradeoff
    score_threshold:      float = 0.0   # Minimum score to include in results

    # ---- SLA ----
    latency_warn_ms:  int = 80    # Warn if request takes >80ms
    latency_error_ms: int = 150   # Error if request takes >150ms

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    lru_cache ensures we only parse env vars once.
    """
    return Settings()