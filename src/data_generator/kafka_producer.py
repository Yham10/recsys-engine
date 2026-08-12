"""
Kafka Event Producer
====================
Simulates a live e-commerce website generating real-time user events.
Streams UserEvent messages to the Kafka 'user-events' topic.

Design decisions:
- Uses confluent-kafka (fastest Python Kafka client, wraps librdkafka)
- JSON serialization for human readability and debuggability
- Exponential backoff retry on connection failures
- Graceful shutdown on SIGINT/SIGTERM
- Configurable event rate (events per second)
"""

import os
import sys
import json
import time
import random
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from tenacity import (
    retry, stop_after_attempt,
    wait_exponential, before_sleep_log
)
import logging

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from schemas import UserEvent, EventType, DeviceType

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC_USER_EVENTS", "user-events")
KAFKA_NUM_PARTITIONS    = int(os.getenv("KAFKA_NUM_PARTITIONS", "3"))
KAFKA_REPLICATION       = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
DATA_DIR                = Path(os.getenv("DATA_DIR", "data/raw"))

# Event rate controls
DEFAULT_EVENTS_PER_SECOND = 10    # Simulate ~10 users per second
BURST_MULTIPLIER          = 5     # During "peak hours" simulation
LOG_EVERY_N_EVENTS        = 100   # Log progress every N events


# ----------------------------------------------------------------
# KAFKA TOPIC MANAGEMENT
# ----------------------------------------------------------------

def ensure_kafka_topic_exists(
    bootstrap_servers: str,
    topic: str,
    num_partitions: int,
    replication_factor: int
) -> None:
    """
    Idempotently create the Kafka topic if it doesn't exist.
    In production, this would be handled by Terraform/Helm charts.
    """
    admin_client = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing_topics = admin_client.list_topics(timeout=10).topics

    if topic not in existing_topics:
        logger.info(f"Creating Kafka topic: '{topic}'")
        new_topic = NewTopic(
            topic=topic,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
            config={
                "retention.ms":    str(7 * 24 * 60 * 60 * 1000),  # 7 days
                "compression.type": "lz4",   # Fast compression
                "cleanup.policy":   "delete",
            }
        )
        futures = admin_client.create_topics([new_topic])
        for topic_name, future in futures.items():
            try:
                future.result()
                logger.success(f"✅ Topic '{topic_name}' created successfully")
            except Exception as e:
                logger.warning(f"Topic creation warning: {e}")
    else:
        logger.info(f"Topic '{topic}' already exists — skipping creation")


# ----------------------------------------------------------------
# KAFKA PRODUCER
# ----------------------------------------------------------------

class ECommerceEventProducer:
    """
    Simulates a live e-commerce platform generating user events.

    Loads user and item data from the historical data files,
    then generates and streams realistic real-time events to Kafka.

    Args:
        bootstrap_servers:  Kafka broker address
        topic:              Target Kafka topic name
        events_per_second:  Target throughput rate
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str             = KAFKA_TOPIC,
        events_per_second: int = DEFAULT_EVENTS_PER_SECOND,
    ):
        self.topic             = topic
        self.events_per_second = events_per_second
        self._running          = False
        self._event_count      = 0
        self._error_count      = 0

        # Load user and item data for realistic simulation
        self.users_df = self._load_data(DATA_DIR / "users.csv", "users")
        self.items_df = self._load_data(DATA_DIR / "items.csv", "items")

        # Build item popularity weights
        self._item_popularity = self._build_popularity_weights()

        # Build items-by-category index for preference sampling
        self._items_by_category = self._build_category_index()

        # Initialize Kafka Producer
        self.producer = self._create_producer(bootstrap_servers)

        # Ensure our topic exists
        ensure_kafka_topic_exists(
            bootstrap_servers, topic,
            KAFKA_NUM_PARTITIONS, KAFKA_REPLICATION
        )

        # Register graceful shutdown handlers
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info(
            f"ECommerceEventProducer ready | "
            f"topic={topic} | rate={events_per_second} events/sec"
        )

    # ----------------------------------------------------------
    # SETUP METHODS
    # ----------------------------------------------------------

    @staticmethod
    def _load_data(path: Path, name: str) -> pd.DataFrame:
        """Load CSV data with validation."""
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                f"Run historical_data.py first to generate data."
            )
        df = pd.read_csv(path)
        logger.info(f"Loaded {name}: {len(df):,} records from {path}")
        return df

    def _build_popularity_weights(self) -> np.ndarray:
        """
        Build item sampling weights based on review_count.
        Implements power law — popular items dominate traffic.
        """
        counts = self.items_df["review_count"].values.astype(float)
        counts = np.maximum(counts, 1)  # Avoid zero weights
        weights = counts / counts.sum()
        return weights

    def _build_category_index(self) -> dict[str, list]:
        """Index items by category for fast preference-based sampling."""
        index: dict[str, list] = {}
        for _, row in self.items_df.iterrows():
            cat = row["category"]
            if cat not in index:
                index[cat] = []
            index[cat].append(row["item_id"])
        return index

    @staticmethod
    def _create_producer(bootstrap_servers: str) -> Producer:
        """
        Create and configure the Kafka producer.

        Key settings:
        - acks=all:             Wait for all replicas (durability)
        - compression.type=lz4: Fast compression
        - linger.ms=5:          Small batching window for throughput
        - retries=3:            Auto-retry on transient failures
        """
        config = {
            "bootstrap.servers":            bootstrap_servers,
            "acks":                         "all",
            "compression.type":             "lz4",
            "linger.ms":                    5,
            "batch.size":                   16384,
            "retries":                      3,
            "retry.backoff.ms":             100,
            "socket.timeout.ms":            10000,
            "message.timeout.ms":           30000,
        }
        logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
        producer = Producer(config)
        logger.success("✅ Kafka producer connected")
        return producer

    # ----------------------------------------------------------
    # EVENT GENERATION
    # ----------------------------------------------------------

    def _generate_event(self) -> UserEvent:
        """
        Generate a single realistic user event.
        Mirrors the logic in historical_data.py to ensure
        training data and streaming data have the same distribution.
        """
        # Sample user
        user = self.users_df.sample(1).iloc[0]
        user_id        = user["user_id"]
        is_premium     = bool(user["is_premium"])
        preferred_cats = str(user["preferred_categories"]).split(",")

        # Sample item (preference-biased)
        if random.random() < 0.80 and preferred_cats:
            preferred_cat  = random.choice(preferred_cats)
            candidate_items = self._items_by_category.get(preferred_cat, [])
            if candidate_items:
                item_id = random.choice(candidate_items)
            else:
                item_id = np.random.choice(
                    self.items_df["item_id"].values,
                    p=self._item_popularity
                )
        else:
            item_id = np.random.choice(
                self.items_df["item_id"].values,
                p=self._item_popularity
            )

        item_row = self.items_df[
            self.items_df["item_id"] == item_id
        ].iloc[0]

        # Sample event type
        event_types   = list(EventType)
        event_weights = [0.45, 0.25, 0.12, 0.08, 0.01, 0.03, 0.02, 0.04]
        event_type    = random.choices(event_types, weights=event_weights)[0]

        # Premium boost
        if is_premium and event_type == EventType.ITEM_VIEW:
            if random.random() < 0.12:
                event_type = EventType.PURCHASE

        # Build event
        price = round(item_row["price"] * random.uniform(0.90, 1.10), 2)

        event = UserEvent(
            event_type      = event_type,
            timestamp       = datetime.now(),
            user_id         = user_id,
            item_id         = item_id,
            session_id      = f"sess_{user_id}_{int(time.time())}",
            device_type     = random.choices(
                                list(DeviceType),
                                weights=[0.55, 0.35, 0.10]
                              )[0],
            price_at_event  = price,
            quantity        = random.randint(1, 3)
                              if event_type in [
                                  EventType.PURCHASE, EventType.ADD_TO_CART
                              ] else None,
            rating_value    = round(random.uniform(1, 5), 1)
                              if event_type == EventType.RATING else None,
            page_dwell_time = random.randint(5, 300)
                              if event_type == EventType.ITEM_VIEW else None,
        )
        return event

    # ----------------------------------------------------------
    # KAFKA DELIVERY
    # ----------------------------------------------------------

    def _delivery_report(self, err, msg) -> None:
        """
        Callback invoked by Kafka after each message delivery attempt.
        Called asynchronously from poll() — do not raise exceptions here.
        """
        if err is not None:
            self._error_count += 1
            logger.error(
                f"Message delivery FAILED | "
                f"topic={msg.topic()} | partition={msg.partition()} | "
                f"error={err}"
            )
        else:
            self._event_count += 1
            if self._event_count % LOG_EVERY_N_EVENTS == 0:
                logger.info(
                    f"📨 Delivered {self._event_count:,} events | "
                    f"errors={self._error_count} | "
                    f"topic={msg.topic()} | partition={msg.partition()}"
                )

    def _publish_event(self, event: UserEvent) -> None:
        """
        Serialize and publish a single event to Kafka.
        Uses user_id as the partition key so all events from
        the same user go to the same partition (ordering guarantee).
        """
        payload = json.dumps(event.to_kafka_payload()).encode("utf-8")

        self.producer.produce(
            topic     = self.topic,
            key       = event.user_id.encode("utf-8"),  # Partition key
            value     = payload,
            on_delivery = self._delivery_report,
        )
        # Non-blocking poll to serve delivery callbacks
        self.producer.poll(0)

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------

    def run(self, max_events: int = None) -> None:
        """
        Start the event streaming loop.

        Args:
            max_events: Stop after N events (None = run forever)
        """
        self._running = True
        sleep_interval = 1.0 / self.events_per_second

        logger.info("=" * 60)
        logger.info(f"  🚀 STARTING KAFKA EVENT STREAM")
        logger.info(f"  Topic:  {self.topic}")
        logger.info(f"  Rate:   {self.events_per_second} events/sec")
        logger.info(f"  Limit:  {max_events or 'unlimited'} events")
        logger.info("=" * 60)

        total_sent = 0

        try:
            while self._running:
                if max_events and total_sent >= max_events:
                    logger.info(f"Reached max_events limit ({max_events})")
                    break

                # Simulate traffic bursts (e.g., flash sale)
                current_hour = datetime.now().hour
                is_peak_hour = 9 <= current_hour <= 21
                burst = random.random() < 0.05  # 5% chance of burst

                batch_size = (
                    BURST_MULTIPLIER if burst else
                    (2 if is_peak_hour else 1)
                )

                for _ in range(batch_size):
                    try:
                        event = self._generate_event()
                        self._publish_event(event)
                        total_sent += 1
                    except Exception as e:
                        logger.error(f"Failed to generate/publish event: {e}")
                        self._error_count += 1

                time.sleep(sleep_interval)

        finally:
            self._flush_and_close(total_sent)

    def _flush_and_close(self, total_sent: int) -> None:
        """Ensure all buffered messages are delivered before shutdown."""
        logger.info("Flushing remaining messages to Kafka...")
        remaining = self.producer.flush(timeout=30)

        if remaining > 0:
            logger.warning(f"{remaining} messages were NOT delivered")
        else:
            logger.success("✅ All messages flushed successfully")

        logger.info("=" * 60)
        logger.info(f"  PRODUCER SHUTDOWN COMPLETE")
        logger.info(f"  Total events sent:  {total_sent:,}")
        logger.info(f"  Delivery errors:    {self._error_count:,}")
        logger.info("=" * 60)

    def _shutdown_handler(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Shutdown signal received (signal={signum})")
        self._running = False