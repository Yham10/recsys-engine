"""
DAG: feature_materialization_pipeline
======================================
Lightweight pipeline that refreshes the online feature store
every 6 hours without retraining the model.

Purpose:
    Between daily training runs, user behavior keeps changing.
    A user who bought something 3 hours ago should have updated
    features when they visit again — without waiting for the
    next full training cycle.

Pipeline:
    check_new_data → materialize_incremental → verify_redis

Run frequency: Every 6 hours (0 */6 * * *)
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

from tasks.feature_tasks import (
    task_materialize_incremental,
    task_verify_redis,
)
from tasks.quality_tasks import task_check_data_freshness


DEFAULT_ARGS = {
    "owner":            "ml-platform",
    "depends_on_past":  False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}


with DAG(
    dag_id            = "feature_materialization_pipeline",
    description       = (
        "Incremental feature refresh every 6 hours. "
        "Keeps Redis online store up-to-date between training runs."
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = "0 */6 * * *",  # Every 6 hours
    start_date        = datetime(2024, 1, 1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["features", "materialization", "6h", "recsys"],
    doc_md            = __doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    # ── Check data freshness before materializing ────────────────
    check_freshness = PythonOperator(
        task_id         = "check_data_freshness",
        python_callable = task_check_data_freshness,
        doc_md          = "Verify data sources are not stale",
    )

    # ── Incremental materialization ──────────────────────────────
    materialize_inc = PythonOperator(
        task_id         = "materialize_incremental",
        python_callable = task_materialize_incremental,
        doc_md          = (
            "Push only new features to Redis "
            "(more efficient than full materialization)"
        ),
    )

    # ── Verify Redis ─────────────────────────────────────────────
    verify = PythonOperator(
        task_id         = "verify_redis_populated",
        python_callable = task_verify_redis,
        doc_md          = "Assert Redis key count > 0",
    )

    end = EmptyOperator(task_id="end")

    start >> check_freshness >> materialize_inc >> verify >> end