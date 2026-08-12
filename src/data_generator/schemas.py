"""
Event Schemas
=============
Pydantic models defining the structure of every event type in our system.
These schemas act as the contract between the data generator,
Kafka, and the feature engineering pipeline.

Think of these as your API contracts — if a field changes here,
it breaks downstream. That's intentional and good.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import uuid


# ----------------------------------------------------------------
# ENUMS
# ----------------------------------------------------------------

class EventType(str, Enum):
    """All possible user interaction event types."""
    PAGE_VIEW    = "page_view"
    ITEM_VIEW    = "item_view"
    SEARCH       = "search"
    ADD_TO_CART  = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PURCHASE     = "purchase"
    RATING       = "rating"
    WISHLIST_ADD = "wishlist_add"


class ItemCategory(str, Enum):
    """Product categories in our simulated e-commerce store."""
    ELECTRONICS   = "electronics"
    CLOTHING      = "clothing"
    BOOKS         = "books"
    HOME_GARDEN   = "home_garden"
    SPORTS        = "sports"
    BEAUTY        = "beauty"
    TOYS          = "toys"
    FOOD          = "food"
    AUTOMOTIVE    = "automotive"
    JEWELRY       = "jewelry"


class DeviceType(str, Enum):
    """Device types for user sessions."""
    MOBILE  = "mobile"
    DESKTOP = "desktop"
    TABLET  = "tablet"


# ----------------------------------------------------------------
# CORE DATA MODELS
# ----------------------------------------------------------------

class UserProfile(BaseModel):
    """
    Represents a user in our system.
    Generated once and reused across all events.
    """
    user_id:          str
    age:              int = Field(ge=18, le=80)
    gender:           str = Field(pattern="^(M|F|Other)$")
    country:          str
    preferred_categories: list[ItemCategory]
    account_age_days: int = Field(ge=0)
    is_premium:       bool

    class Config:
        use_enum_values = True


class ItemProfile(BaseModel):
    """
    Represents a product/item in our catalog.
    Generated once and reused across all events.
    """
    item_id:      str
    item_name:    str
    category:     ItemCategory
    subcategory:  str
    price:        float = Field(gt=0)
    avg_rating:   float = Field(ge=0, le=5)
    review_count: int   = Field(ge=0)
    brand:        str
    is_available: bool

    class Config:
        use_enum_values = True


class UserEvent(BaseModel):
    """
    A single user interaction event.
    This is the core message published to Kafka.
    Every field here becomes a potential feature.
    """
    event_id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:      EventType
    timestamp:       datetime
    user_id:         str
    item_id:         Optional[str] = None         # None for page_view/search
    session_id:      str
    device_type:     DeviceType
    price_at_event:  Optional[float] = None       # Price when event happened
    quantity:        Optional[int]   = None       # For purchase/cart events
    rating_value:    Optional[float] = None       # For rating events (1-5)
    search_query:    Optional[str]   = None       # For search events
    page_dwell_time: Optional[int]   = None       # Seconds spent on page

    # Derived engagement weight (used as implicit feedback label)
    engagement_weight: float = Field(default=1.0)

    @field_validator("engagement_weight", mode="before")
    @classmethod
    def set_engagement_weight(cls, v, info):
        """
        Assign engagement weight based on event type.
        Purchase > Add to Cart > Rating > Wishlist > View
        This becomes our implicit feedback signal for training.
        """
        weights = {
            EventType.PAGE_VIEW:        0.1,
            EventType.ITEM_VIEW:        0.3,
            EventType.SEARCH:           0.2,
            EventType.ADD_TO_CART:      0.7,
            EventType.REMOVE_FROM_CART: -0.2,
            EventType.PURCHASE:         1.0,
            EventType.RATING:           0.8,
            EventType.WISHLIST_ADD:     0.5,
        }
        event_type = info.data.get("event_type")
        if event_type:
            return weights.get(event_type, 1.0)
        return v

    class Config:
        use_enum_values = True

    def to_kafka_payload(self) -> dict:
        """
        Serialize event to a flat dict for Kafka.
        Timestamps are converted to ISO strings for JSON serialization.
        """
        payload = self.model_dump()
        payload["timestamp"] = self.timestamp.isoformat()
        return payload