"""
DAG: daily_training_pipeline
=============================
The master ML pipeline — runs every day at 2:00 AM.

Full pipeline:
    1. Load enriched features → PostgreSQL
    2. Apply Feast registry (schema sync)
    3. Materialize features   → Redis
    4. Verify Redis populated
    5. Build training dataset (point-in-time correct)
    6. Train Two-Tower model
    7. Evaluate against quality gates
    8. Register model in MLflow Registry
    9. Smoke test live API

On failure:
    - Task retries up to 2 times with 5-minute delay
    - Failure email/alert sent (configure email in Airflow)
    - Old production model remains untouched

Manual trigger with custom config:
    {
        "sample_size": 50000,
        "embedding_dim": 64,
        "epochs": 20,
        "min_auc": 0.70
    }
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# import sys
# import os
# from pathlib import Path

# sys.path.insert(
#     0,
#     str(Path(__file__).parent.parent / "tasks")
# )

from tasks.feature_tasks import (
    task_load_features_to_postgres,
    task_feast_apply,
    task_materialize_features,
    task_verify_redis,
)
from tasks.training_tasks import (
    task_build_training_dataset,
    task_train_model,
    task_evaluate_and_gate,
    task_register_model,
    task_smoke_test,
)


# ----------------------------------------------------------------
# DEFAULT ARGS
# Applied to every task unless overridden
# ----------------------------------------------------------------

DEFAULT_ARGS = {
    "owner":            "ml-platform",
    "depends_on_past":  False,
    "email":            ["ml-alerts@recsys.local"],
    "email_on_failure": False,  # Set True when email is configured
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
}


# ----------------------------------------------------------------
# DAG DEFINITION
# ----------------------------------------------------------------

with DAG(
    dag_id            = "daily_training_pipeline",
    description       = (
        "Daily ML pipeline: feature loading → training → "
        "evaluation → registration → smoke test"
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = "0 2 * * *",    # Every day at 2:00 AM
    start_date        = datetime(2024, 1, 1),
    catchup           = False,           # Don't backfill missed runs
    max_active_runs   = 1,              # Only one run at a time
    tags              = ["ml", "training", "daily", "recsys"],
    doc_md            = __doc__,
) as dag:

    # ── Start marker ────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Task 1: Load features to PostgreSQL ─────────────────────
    load_features = PythonOperator(
        task_id         = "load_features_to_postgres",
        python_callable = task_load_features_to_postgres,
        doc_md          = "Load enriched feature CSVs into PostgreSQL",
    )

    # ── Task 2: Apply Feast registry ────────────────────────────
    feast_apply = PythonOperator(
        task_id         = "feast_apply_registry",
        python_callable = task_feast_apply,
        doc_md          = "Sync Feast feature definitions to registry",
    )

    # ── Task 3: Materialize features → Redis ────────────────────
    materialize = PythonOperator(
        task_id         = "materialize_features_to_redis",
        python_callable = task_materialize_features,
        doc_md          = "Push offline features to Redis online store",
    )

    # ── Task 4: Verify Redis ─────────────────────────────────────
    verify_redis = PythonOperator(
        task_id         = "verify_redis_populated",
        python_callable = task_verify_redis,
        doc_md          = "Assert Redis contains feature keys",
    )

    # ── Task 5: Build training dataset ──────────────────────────
    build_dataset = PythonOperator(
        task_id         = "build_training_dataset",
        python_callable = task_build_training_dataset,
        doc_md          = (
            "Build point-in-time correct train/val/test splits "
            "using Feast historical features"
        ),
    )

    # ── Task 6: Train model ──────────────────────────────────────
    train_model = PythonOperator(
        task_id          = "train_model",
        python_callable  = task_train_model,
        execution_timeout = timedelta(hours=3),
        doc_md           = "Train Two-Tower neural network + log to MLflow",
    )

    # ── Task 7: Evaluate & gate ──────────────────────────────────
    evaluate = PythonOperator(
        task_id         = "evaluate_and_gate",
        python_callable = task_evaluate_and_gate,
        doc_md          = (
            "Check model meets quality thresholds "
            "(test_auc >= 0.70, recall@10 >= 0.05)"
        ),
    )

    # ── Task 8: Register model ───────────────────────────────────
    register = PythonOperator(
        task_id         = "register_model",
        python_callable = task_register_model,
        doc_md          = "Promote model to Production in MLflow Registry",
    )

    # ── Task 9: Smoke test ───────────────────────────────────────
    smoke_test = PythonOperator(
        task_id         = "smoke_test",
        python_callable = task_smoke_test,
        doc_md          = "Verify live API returns valid recommendations",
    )

    # ── End marker ───────────────────────────────────────────────
    end = EmptyOperator(task_id="end")

    # ── Dependencies (the DAG graph) ─────────────────────────────
    (
        start
        >> load_features
        >> feast_apply
        >> materialize
        >> verify_redis
        >> build_dataset
        >> train_model
        >> evaluate
        >> register
        >> smoke_test
        >> end
    )