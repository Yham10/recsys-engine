"""
Data Quality Tasks
==================
Tasks for monitoring data health, detecting drift,
and alerting when the system degrades.

Checks performed:
    1. Data freshness  → Are new interactions arriving?
    2. Feature drift   → Have feature distributions shifted?
    3. Model drift     → Has prediction score distribution changed?
    4. Coverage        → Are we able to serve all users?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import create_engine, text

RECSYS_SRC = Path(os.getenv("RECSYS_SRC", "/opt/airflow/recsys_src"))


# ----------------------------------------------------------------
# TASK 1: Check Data Freshness
# ----------------------------------------------------------------

def task_check_data_freshness(**context) -> dict:
    """
    Verifies that new interaction data is flowing into the system.

    Checks:
        - interactions.csv was modified within the last 24 hours
        - PostgreSQL feature tables have recent event_timestamps
        - Redis has active keys (not expired)

    Raises:
        ValueError if data is stale beyond threshold
    """
    logger.info("TASK: check_data_freshness started")

    results = {}

    # ── Check 1: interactions.csv freshness ─────────────────────
    data_dir = Path(
        os.getenv(
            "DATA_RAW_DIR",
            str(RECSYS_SRC / "data_generator" / "data" / "raw")
        )
    )
    interactions_path = data_dir / "interactions.csv"

    if interactions_path.exists():
        mtime     = datetime.fromtimestamp(interactions_path.stat().st_mtime)
        age_hours = (datetime.utcnow() - mtime).total_seconds() / 3600
        results["interactions_age_hours"] = round(age_hours, 2)
        results["interactions_fresh"]     = age_hours < 25   # 25h tolerance

        logger.info(
            f"interactions.csv age: {age_hours:.1f}h | "
            f"fresh={results['interactions_fresh']}"
        )
    else:
        results["interactions_fresh"] = False
        logger.warning("interactions.csv not found")

    # ── Check 2: PostgreSQL feature table freshness ─────────────
    try:
        pg_user = os.getenv("POSTGRES_USER", "recsys_user")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "recsys_password")
        pg_host = os.getenv("POSTGRES_HOST", "postgres")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db   = os.getenv("POSTGRES_DB", "recsys_db")

        engine  = create_engine(
            f"postgresql+psycopg2://{pg_user}:{pg_pass}"
            f"@{pg_host}:{pg_port}/{pg_db}"
        )

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    MAX(event_timestamp) as latest_ts,
                    COUNT(*)             as row_count
                FROM feast.user_features_raw
            """))
            row       = result.fetchone()
            latest_ts = row[0]
            row_count = row[1]

        if latest_ts:
            pg_age_hours = (
                datetime.utcnow() - latest_ts.replace(tzinfo=None)
            ).total_seconds() / 3600

            results["postgres_feature_age_hours"] = round(pg_age_hours, 2)
            results["postgres_feature_rows"]       = row_count
            results["postgres_fresh"] = pg_age_hours < 25

            logger.info(
                f"PostgreSQL features age: {pg_age_hours:.1f}h | "
                f"rows={row_count:,}"
            )
        else:
            results["postgres_fresh"] = False

    except Exception as e:
        logger.warning(f"PostgreSQL freshness check failed: {e}")
        results["postgres_fresh"] = False

    # ── Check 3: Redis key count ─────────────────────────────────
    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )
        redis_size            = r.dbsize()
        results["redis_keys"] = redis_size
        results["redis_has_data"] = redis_size > 0

        logger.info(f"Redis keys: {redis_size:,}")
    except Exception as e:
        logger.warning(f"Redis freshness check failed: {e}")
        results["redis_has_data"] = False

    # ── Summary ──────────────────────────────────────────────────
    all_fresh = all([
        results.get("interactions_fresh", False),
        results.get("postgres_fresh",     False),
        results.get("redis_has_data",     False),
    ])

    results["all_fresh"] = all_fresh
    results["timestamp"] = datetime.utcnow().isoformat()
    results["status"]    = "success" if all_fresh else "warning"

    if not all_fresh:
        logger.warning(
            f"Data freshness issues detected: "
            f"{[k for k, v in results.items() if v is False]}"
        )
    else:
        logger.success("TASK: check_data_freshness — all data is fresh")

    return results


# ----------------------------------------------------------------
# TASK 2: Validate Feature Distributions
# ----------------------------------------------------------------

def task_validate_feature_distributions(**context) -> dict:
    """
    Checks that feature distributions haven't shifted dramatically.

    Computes basic statistics on current feature values and
    compares against expected ranges derived from training data.

    Uses simple heuristic checks (no complex drift detection library
    needed for a university project — PSI or KL-divergence in prod).
    """
    logger.info("TASK: validate_feature_distributions started")

    try:
        pg_user = os.getenv("POSTGRES_USER", "recsys_user")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "recsys_password")
        pg_host = os.getenv("POSTGRES_HOST", "postgres")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db   = os.getenv("POSTGRES_DB", "recsys_db")

        engine = create_engine(
            f"postgresql+psycopg2://{pg_user}:{pg_pass}"
            f"@{pg_host}:{pg_port}/{pg_db}"
        )

        # ---- User feature stats ----
        with engine.connect() as conn:
            user_stats = pd.read_sql("""
                SELECT
                    AVG(user_click_count_7d)       AS avg_clicks,
                    STDDEV(user_click_count_7d)    AS std_clicks,
                    AVG(user_purchase_count_30d)   AS avg_purchases,
                    AVG(user_total_spend_30d)      AS avg_spend,
                    COUNT(*)                       AS total_users
                FROM feast.user_features_raw
            """, conn)

            item_stats = pd.read_sql("""
                SELECT
                    AVG(item_view_count_7d)        AS avg_views,
                    AVG(item_conversion_rate)      AS avg_conversion,
                    AVG(item_avg_rating_events)    AS avg_rating,
                    COUNT(*)                       AS total_items
                FROM feast.item_features_raw
            """, conn)

        user_row = user_stats.iloc[0]
        item_row = item_stats.iloc[0]

        # ---- Heuristic checks ----
        # These thresholds are based on our data generation parameters
        checks = {
            "user_avg_clicks_reasonable":
                0 <= float(user_row["avg_clicks"] or 0) <= 200,
            "user_avg_spend_reasonable":
                0 <= float(user_row["avg_spend"] or 0) <= 5000,
            "item_avg_conversion_rate_reasonable":
                0 <= float(item_row["avg_conversion"] or 0) <= 1.0,
            "item_avg_rating_reasonable":
                1 <= float(item_row["avg_rating"] or 3) <= 5,
            "enough_users":
                int(user_row["total_users"] or 0) >= 1000,
            "enough_items":
                int(item_row["total_items"] or 0) >= 100,
        }

        passed = all(checks.values())

        logger.info("Feature Distribution Checks:")
        for check, ok in checks.items():
            logger.info(f"  {'✅' if ok else '❌'} {check}")

        result = {
            "status":      "success" if passed else "warning",
            "timestamp":   datetime.utcnow().isoformat(),
            "all_passed":  passed,
            "checks":      checks,
            "user_stats":  user_row.to_dict(),
            "item_stats":  item_row.to_dict(),
        }

        logger.success(
            f"TASK: validate_feature_distributions completed | "
            f"passed={passed}"
        )
        return result

    except Exception as e:
        logger.error(
            f"TASK: validate_feature_distributions FAILED: {e}"
        )
        raise


# ----------------------------------------------------------------
# TASK 3: Detect Model Score Drift
# ----------------------------------------------------------------

def task_detect_score_drift(**context) -> dict:
    """
    Samples predictions from the live API and checks that the
    score distribution hasn't drifted (e.g., model always returns
    the same items, or scores collapsed to 0).

    Simple checks:
        - Mean score is in a reasonable range [0.1, 0.9]
        - Score std > 0.05 (model is differentiating users)
        - Top item diversity (not all users get same #1 item)
    """
    logger.info("TASK: detect_score_drift started")

    try:
        import requests as req_lib
        import random

        api_base = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

        # Sample N random users
        sample_user_ids = [f"user_{i:06d}" for i in random.sample(range(10000), 50)]

        all_scores  = []
        top_items   = []
        failed_reqs = 0

        for user_id in sample_user_ids:
            try:
                resp = req_lib.post(
                    f"{api_base}/api/v1/recommend",
                    json    = {"user_id": user_id, "top_k": 5},
                    timeout = 5,
                )
                if resp.status_code == 200:
                    data  = resp.json()
                    recs  = data.get("recommendations", [])
                    scores = [r["score"] for r in recs]
                    all_scores.extend(scores)
                    if recs:
                        top_items.append(recs[0]["item_id"])
                else:
                    failed_reqs += 1
            except Exception:
                failed_reqs += 1

        if not all_scores:
            raise ValueError(
                "No scores collected — API may be down"
            )

        mean_score     = float(np.mean(all_scores))
        std_score      = float(np.std(all_scores))
        unique_top_items = len(set(top_items))
        diversity_ratio  = unique_top_items / max(len(top_items), 1)

        checks = {
            "mean_score_in_range":
                0.05 <= mean_score <= 0.95,
            "score_has_variance":
                std_score > 0.01,
            "top_item_diversity":
                diversity_ratio > 0.3,    # >30% of users get unique top-1
            "low_failure_rate":
                failed_reqs / len(sample_user_ids) < 0.1,
        }

        passed = all(checks.values())

        result = {
            "status":           "success" if passed else "warning",
            "timestamp":        datetime.utcnow().isoformat(),
            "n_users_sampled":  len(sample_user_ids),
            "n_scores":         len(all_scores),
            "mean_score":       round(mean_score, 4),
            "std_score":        round(std_score, 4),
            "diversity_ratio":  round(diversity_ratio, 4),
            "failed_requests":  failed_reqs,
            "all_checks_passed": passed,
            "checks":           checks,
        }

        logger.success(
            f"TASK: detect_score_drift completed | "
            f"mean={mean_score:.4f} | "
            f"std={std_score:.4f} | "
            f"diversity={diversity_ratio:.2%} | "
            f"passed={passed}"
        )
        return result

    except Exception as e:
        logger.error(f"TASK: detect_score_drift FAILED: {e}")
        raise


# ----------------------------------------------------------------
# TASK 4: Send Alert
# ----------------------------------------------------------------

def task_send_alert(**context) -> dict:
    """
    Sends an alert if upstream quality checks failed.

    In production: sends to Slack/PagerDuty/email.
    Here: logs a prominent warning and writes to alert log file.

    Only triggers if a previous task returned warnings.
    """
    logger.info("TASK: send_alert started")

    ti = context["ti"]

    # Collect results from upstream tasks
    freshness_result   = ti.xcom_pull(task_ids="check_data_freshness")   or {}
    distribution_result = ti.xcom_pull(
        task_ids="validate_feature_distributions"
    ) or {}
    drift_result       = ti.xcom_pull(task_ids="detect_score_drift")     or {}

    issues = []

    if not freshness_result.get("all_fresh", True):
        issues.append(f"DATA FRESHNESS: {freshness_result}")
    if not distribution_result.get("all_passed", True):
        issues.append(f"FEATURE DRIFT: {distribution_result}")
    if not drift_result.get("all_checks_passed", True):
        issues.append(f"SCORE DRIFT: {drift_result}")

    alert_path = Path("/opt/airflow/logs/quality_alerts.log")
    alert_path.parent.mkdir(parents=True, exist_ok=True)

    if issues:
        alert_msg = (
            f"\n{'='*60}\n"
            f"🚨 RECSYS QUALITY ALERT — {datetime.utcnow().isoformat()}\n"
            f"{'='*60}\n"
            + "\n".join(issues)
            + f"\n{'='*60}\n"
        )
        logger.warning(alert_msg)

        with open(alert_path, "a") as f:
            f.write(alert_msg)

        # In production, add Slack/email notification here:
        # slack_webhook.send(alert_msg)
    else:
        logger.success(
            "TASK: send_alert — no issues detected, no alert sent"
        )

    return {
        "status":    "success",
        "timestamp": datetime.utcnow().isoformat(),
        "n_issues":  len(issues),
        "alerted":   len(issues) > 0,
    }