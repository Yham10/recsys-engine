"""
DAG: data_quality_pipeline
============================
Daily monitoring pipeline that checks the health of our entire
system and alerts if anything looks wrong.

Runs at 1:00 AM — one hour BEFORE the training pipeline (2:00 AM)
so we know the data quality before we start training on it.

Checks:
    1. Data freshness      (is new data arriving?)
    2. Feature distributions (have features drifted?)
    3. Model score drift   (is the model still differentiating?)
    4. Send alert if any checks fail

This DAG does NOT fail on quality warnings — it logs them.
Only critical failures (API down, DB unreachable) raise exceptions.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# import sys
# from pathlib import Path

# sys.path.insert(
#     0,
#     str(Path(__file__).parent.parent / "tasks")
# )

from tasks.quality_tasks import (
    task_check_data_freshness,
    task_validate_feature_distributions,
    task_detect_score_drift,
    task_send_alert,
)


DEFAULT_ARGS = {
    "owner":            "ml-platform",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=45),
}


with DAG(
    dag_id            = "data_quality_pipeline",
    description       = (
        "Daily data quality monitoring: freshness checks, "
        "feature distribution validation, model score drift detection."
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = "0 1 * * *",    # Every day at 1:00 AM
    start_date        = datetime(2024, 1, 1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["monitoring", "quality", "daily", "recsys"],
    doc_md            = __doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    # ── Check 1: Data freshness ──────────────────────────────────
    freshness = PythonOperator(
        task_id         = "check_data_freshness",
        python_callable = task_check_data_freshness,
        doc_md          = "Check interactions.csv, PostgreSQL, Redis freshness",
    )

    # ── Check 2: Feature distributions ──────────────────────────
    distributions = PythonOperator(
        task_id         = "validate_feature_distributions",
        python_callable = task_validate_feature_distributions,
        doc_md          = "Validate feature stats are within expected ranges",
    )

    # ── Check 3: Model score drift ───────────────────────────────
    score_drift = PythonOperator(
        task_id         = "detect_score_drift",
        python_callable = task_detect_score_drift,
        doc_md          = "Sample API predictions and check score distribution",
    )

    # ── Alert if issues found ────────────────────────────────────
    alert = PythonOperator(
        task_id         = "send_alert_if_degraded",
        python_callable = task_send_alert,
        trigger_rule    = "all_done",  # Run even if upstream tasks warn
        doc_md          = "Send alert if any quality checks failed",
    )

    end = EmptyOperator(task_id="end")

    # ── Checks run in parallel, then alert ──────────────────────
    start >> [freshness, distributions, score_drift] >> alert >> end