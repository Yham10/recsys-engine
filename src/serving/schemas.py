"""
API Schemas
===========
Pydantic models defining the request/response contracts.
These are the public API — changing them is a breaking change.

Versioning strategy: /api/v1/recommend
When we change the schema, we add /api/v2/ and deprecate v1.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


# ----------------------------------------------------------------
# REQUEST SCHEMAS
# ----------------------------------------------------------------

class RecommendationRequest(BaseModel):
    """
    Request body for the recommendation endpoint.

    Example:
        POST /api/v1/recommend
        {
            "user_id": "user_000123",
            "top_k": 10,
            "exclude_item_ids": ["item_000001", "item_000002"],
            "category_filter": "electronics"
        }
    """
    user_id: str = Field(
        ...,
        description = "Unique user identifier",
        example     = "user_000123",
        min_length  = 1,
        max_length  = 64,
    )
    top_k: int = Field(
        default     = 10,
        ge          = 1,
        le          = 50,
        description = "Number of recommendations to return",
    )
    exclude_item_ids: list[str] = Field(
        default_factory = list,
        description     = "Item IDs to exclude (e.g., already purchased)",
        max_length      = 100,
    )
    category_filter: Optional[str] = Field(
        default     = None,
        description = "Filter results to a specific product category",
    )

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be empty or whitespace")
        return v.strip()


# ----------------------------------------------------------------
# RESPONSE SCHEMAS
# ----------------------------------------------------------------

class RecommendedItem(BaseModel):
    """A single recommended item with metadata and scores."""
    item_id:          str
    rank:             int            = Field(description="1-indexed rank in results")
    score:            float          = Field(description="Model similarity score")
    item_name:        Optional[str]  = None
    category:         Optional[str]  = None
    price:            Optional[float] = None
    avg_rating:       Optional[float] = None
    explanation:      Optional[str]  = None  # Why this item was recommended


class RecommendationResponse(BaseModel):
    """
    Full recommendation API response.

    Includes recommendations, metadata, and latency breakdown
    for monitoring and debugging.
    """
    user_id:          str
    recommendations:  list[RecommendedItem]
    top_k_requested:  int
    top_k_returned:   int
    is_cold_start:    bool = Field(
        description="True if user had no features (new user)"
    )

    # Latency breakdown (milliseconds)
    latency_ms:           float
    feature_latency_ms:   Optional[float] = None
    model_latency_ms:     Optional[float] = None
    ann_latency_ms:       Optional[float] = None

    # Request tracing
    request_id:       str
    served_at:        datetime
    model_version:    str


class HealthResponse(BaseModel):
    """Health check response."""
    status:           str     # "healthy" | "degraded" | "unhealthy"
    version:          str
    environment:      str
    model_loaded:     bool
    faiss_loaded:     bool
    redis_connected:  bool
    uptime_seconds:   float
    checks:           dict[str, bool]


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response."""
    ready:   bool
    reason:  Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error:      str
    detail:     Optional[str] = None
    request_id: Optional[str] = None
    timestamp:  datetime = Field(default_factory=datetime.utcnow)