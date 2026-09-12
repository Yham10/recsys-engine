"""
Real-Time Kafka Consumer
=========================
Reads raw events from Kafka, incrementally updates Redis counters,
and appends events to a streaming CSV for future training.

Design:
  - Atomic Redis increments (HINCRBY/HINCRBYFLOAT)
  - Separate namespace: realtime:user:{id} / realtime:item:{id}
  - 24h TTL approximates rolling windows without Flink/Spark
  - Graceful degradation, manual offset commits, poison-pill safe
"""

import os
import json
import csv
import signal
import sys
from pathlib import Path
from loguru import logger
import redis
from confluent_kafka import Consumer, KafkaError

# ── Configuration ────────────────────────────────────────────────
KAFKA_BROKER  = os.getenv("KAFKA_BROKER", "localhost:29092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "user-events")
KAFKA_GROUP   = os.getenv("KAFKA_GROUP", "recsys-realtime-consumer")
REDIS_HOST    = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB      = int(os.getenv("REDIS_DB", 0))
CSV_PATH      = Path(os.getenv("STREAMING_CSV_PATH", "data/raw/streaming_interactions.csv"))
TTL_SECONDS   = int(os.getenv("REALTIME_TTL", 86400))  # 24 hours

# ── Logging ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# ── Redis Client ─────────────────────────────────────────────────
r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
    decode_responses=True, socket_connect_timeout=2.0, socket_timeout=2.0
)

# ── CSV Schema (matches interactions.csv) ────────────────────────
CSV_HEADER = [
    "event_id", "event_type", "timestamp", "user_id", "item_id",
    "session_id", "device_type", "price_at_event", "quantity",
    "rating_value", "engagement_weight"
]

def ensure_csv_header():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        logger.info(f"✅ Created streaming CSV: {CSV_PATH}")

def append_to_csv(event: dict):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([event.get(col, "") for col in CSV_HEADER])

def update_realtime_features(event: dict):
    # 🔒 FIX: Strip whitespace from all keys to prevent "ghost" Redis keys
    user_id = str(event.get("user_id", "")).strip()
    item_id = str(event.get("item_id", "")).strip()
    event_type = str(event.get("event_type", "")).lower().strip()
    price = float(event.get("price_at_event", 0) or 0)

    if not user_id or not item_id:
        return

    # 🔍 DEBUG: Log exactly what we are processing
    logger.info(f"🔍 Processing Event | user={user_id} | type={event_type} | price={price}")

    # ── User Deltas ──────────────────────────────────────────────
    user_key = f"realtime:user:{user_id}"
    if event_type in ("item_view", "page_view"):
        r.hincrby(user_key, "click_delta", 1)
        logger.debug(f"  → Incremented click_delta for {user_id}")
    elif event_type == "purchase":
        r.hincrby(user_key, "purchase_delta", 1)
        r.hincrbyfloat(user_key, "spend_delta", price)
        logger.debug(f"  → Incremented purchase_delta & spend_delta for {user_id}")

    # ── Item Deltas ──────────────────────────────────────────────
    item_key = f"realtime:item:{item_id}"
    if event_type in ("item_view", "page_view"):
        r.hincrby(item_key, "view_delta", 1)
    elif event_type == "purchase":
        r.hincrby(item_key, "purchase_delta", 1)

    # ── Set TTL only once per key ────────────────────────────────
    if r.ttl(user_key) == -1:
        r.expire(user_key, TTL_SECONDS)
    if r.ttl(item_key) == -1:
        r.expire(item_key, TTL_SECONDS)

def run_consumer():
    ensure_csv_header()
    conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": KAFKA_GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "session.timeout.ms": 6000,
    }
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"🚀 Kafka consumer started | topic={KAFKA_TOPIC} | broker={KAFKA_BROKER}")

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        logger.info("⛔ Shutdown signal received. Closing consumer...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    processed = 0
    try:
        while running:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                update_realtime_features(event)
                append_to_csv(event)
                processed += 1

                if processed % 50 == 0:
                    logger.info(f"📨 Processed {processed} events")
                    consumer.commit()
            except Exception as e:
                logger.error(f"Message processing failed: {e}")
                consumer.commit()  # Avoid poison-pill blocking
    finally:
        consumer.commit()
        consumer.close()
        logger.success("✅ Consumer shut down gracefully")

if __name__ == "__main__":
    run_consumer()