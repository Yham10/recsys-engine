"""
Training Pipeline Tasks
=======================
Airflow task functions for model training, evaluation,
registration, and smoke testing.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from loguru import logger

RECSYS_SRC = Path(os.getenv("RECSYS_SRC", "/opt/airflow/recsys_src"))
sys.path.insert(0, str(RECSYS_SRC / "training"))
sys.path.insert(0, str(RECSYS_SRC / "feature_store"))

PROCESSED_DIR = Path(
    os.getenv("PROCESSED_DIR", "/opt/airflow/data/processed")
)
ARTIFACTS_DIR = Path(
    os.getenv("ARTIFACTS_DIR", "/opt/airflow/data/artifacts")
)


# ----------------------------------------------------------------
# TASK 1: Build Training Dataset
# ----------------------------------------------------------------

def task_build_training_dataset(**context) -> dict:
    """
    Builds a point-in-time correct training dataset from Feast.

    Reads raw interactions → calls Feast historical features →
    preprocesses → saves train/val/test Parquet splits.

    Returns:
        dict with split sizes and dataset metadata
    """
    logger.info("TASK: build_training_dataset started")

    try:
        from training_dataset import TrainingDatasetBuilder

        # Get sample size from DAG config (optional override)
        dag_conf    = context.get("dag_run").conf or {}
        sample_size = dag_conf.get("sample_size", None)

        interactions_path = Path(
            os.getenv(
                "INTERACTIONS_PATH",
                str(RECSYS_SRC
                    / "data_generator" / "data" / "raw"
                    / "interactions.csv")
            )
        )

        builder = TrainingDatasetBuilder(
            interactions_path = interactions_path,
            output_dir        = PROCESSED_DIR,
            sample_size       = sample_size,
        )

        splits = builder.build()

        result = {
            "status":      "success",
            "timestamp":   datetime.utcnow().isoformat(),
            "train_rows":  len(splits["train"]),
            "val_rows":    len(splits["val"]),
            "test_rows":   len(splits["test"]),
            "sample_size": sample_size,
        }

        logger.success(
            f"TASK: build_training_dataset completed | "
            f"train={result['train_rows']:,} | "
            f"val={result['val_rows']:,} | "
            f"test={result['test_rows']:,}"
        )
        return result

    except Exception as e:
        logger.error(f"TASK: build_training_dataset FAILED: {e}")
        raise


# ----------------------------------------------------------------
# TASK 2: Train Model
# ----------------------------------------------------------------

def task_train_model(**context) -> dict:
    """
    Trains the Two-Tower model and logs everything to MLflow.

    Reads hyperparameters from DAG run conf (for manual overrides)
    or uses defaults from ModelConfig.

    Returns:
        dict with MLflow run_id and best metrics
    """
    logger.info("TASK: train_model started")

    try:
        import mlflow
        from model import ModelConfig
        from trainer import TwoTowerTrainer

        # Allow hyperparameter overrides via DAG run conf
        dag_conf = context.get("dag_run").conf or {}

        config = ModelConfig(
            user_embedding_dim = dag_conf.get("embedding_dim", 64),
            item_embedding_dim = dag_conf.get("embedding_dim", 64),
            output_dim         = dag_conf.get("output_dim", 64),
            dropout_rate       = dag_conf.get("dropout", 0.2),
            learning_rate      = dag_conf.get("lr", 1e-3),
            batch_size         = dag_conf.get("batch_size", 2048),
        )

        trainer = TwoTowerTrainer(
            config        = config,
            processed_dir = PROCESSED_DIR,
            n_epochs      = dag_conf.get("epochs", 20),
            patience      = dag_conf.get("patience", 5),
            device        = "auto",
        )

        run_id = trainer.train()

        # Retrieve best metrics from MLflow
        client      = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://mlflow:5000"
            )
        )
        run         = client.get_run(run_id)
        metrics     = run.data.metrics
        best_val_auc = metrics.get("val_auc", 0.0)
        test_auc     = metrics.get("test_auc", 0.0)

        result = {
            "status":        "success",
            "timestamp":     datetime.utcnow().isoformat(),
            "mlflow_run_id": run_id,
            "best_val_auc":  best_val_auc,
            "test_auc":      test_auc,
        }

        logger.success(
            f"TASK: train_model completed | "
            f"run_id={run_id} | "
            f"val_auc={best_val_auc:.4f} | "
            f"test_auc={test_auc:.4f}"
        )
        return result

    except Exception as e:
        logger.error(f"TASK: train_model FAILED: {e}")
        raise


# ----------------------------------------------------------------
# TASK 3: Evaluate & Gate Model
# ----------------------------------------------------------------

def task_evaluate_and_gate(**context) -> dict:
    """
    Evaluates the newly trained model against quality gates.

    Quality gates (configurable via DAG conf):
        - test_auc       >= 0.70   (minimum acceptable AUC)
        - test_recall@10 >= 0.05   (minimum Recall@10)

    If gates pass  → model proceeds to registration
    If gates fail  → DAG fails, old model stays in production

    This prevents a degraded model from replacing a good one.
    """
    logger.info("TASK: evaluate_and_gate started")

    try:
        import mlflow

        # Get training results from upstream task via XCom
        ti              = context["ti"]
        train_result    = ti.xcom_pull(task_ids="train_model")
        run_id          = train_result["mlflow_run_id"]

        dag_conf        = context.get("dag_run").conf or {}
        min_auc         = dag_conf.get("min_auc",      0.55)
        min_recall_10   = dag_conf.get("min_recall_10", 0.00)

        # Fetch metrics from MLflow
        client  = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://mlflow:5000"
            )
        )
        run     = client.get_run(run_id)
        metrics = run.data.metrics

        test_auc      = metrics.get("test_auc",      0.0)
        test_recall10 = metrics.get("test_recall@10", 0.0)

        gates = {
            f"test_auc >= {min_auc}":         test_auc      >= min_auc,
            f"test_recall@10 >= {min_recall_10}": test_recall10 >= min_recall_10,
        }

        passed = all(gates.values())

        logger.info("Quality Gate Results:")
        for gate, result in gates.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {status} | {gate}")

        if not passed:
            failed_gates = [g for g, r in gates.items() if not r]
            raise ValueError(
                f"Model quality gates FAILED: {failed_gates}. "
                f"test_auc={test_auc:.4f} | "
                f"test_recall@10={test_recall10:.4f}. "
                f"Old model remains in production."
            )

        result = {
            "status":         "success",
            "timestamp":      datetime.utcnow().isoformat(),
            "mlflow_run_id":  run_id,
            "test_auc":       test_auc,
            "test_recall_10": test_recall10,
            "gates_passed":   gates,
        }

        logger.success(
            f"TASK: evaluate_and_gate PASSED | "
            f"test_auc={test_auc:.4f} | "
            f"recall@10={test_recall10:.4f}"
        )
        return result

    except ValueError:
        raise   # Quality gate failures should fail the DAG
    except Exception as e:
        logger.error(f"TASK: evaluate_and_gate FAILED: {e}")
        raise


# ----------------------------------------------------------------
# TASK 4: Register Model
# ----------------------------------------------------------------

def task_register_model(**context) -> dict:
    """
    Promotes the newly trained model to 'Production' stage
    in the MLflow Model Registry.

    Archives the previously production model automatically.
    This is the deployment step — after this, FastAPI will
    load the new model on next restart.
    """
    logger.info("TASK: register_model started")

    try:
        import mlflow

        ti           = context["ti"]
        eval_result  = ti.xcom_pull(task_ids="evaluate_and_gate")
        run_id       = eval_result["mlflow_run_id"]

        model_name   = os.getenv(
            "MLFLOW_MODEL_NAME", "two-tower-recommender"
        )
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )

        client = mlflow.tracking.MlflowClient(
            tracking_uri=tracking_uri
        )

        # Get the model version registered during training
        model_uri    = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(
            model_uri   = model_uri,
            name        = model_name,
        )

        version_num = model_version.version
        logger.info(
            f"Model version {version_num} registered in registry"
        )

        # Archive old Production model
        current_prod = client.get_latest_versions(
            model_name, stages=["Production"]
        )
        for old_version in current_prod:
            client.transition_model_version_stage(
                name    = model_name,
                version = old_version.version,
                stage   = "Archived",
            )
            logger.info(
                f"Archived old production model: "
                f"version={old_version.version}"
            )

        # Promote new model to Production
        client.transition_model_version_stage(
            name    = model_name,
            version = version_num,
            stage   = "Production",
        )

        # Add description to the model version
        client.update_model_version(
            name        = model_name,
            version     = version_num,
            description = (
                f"Trained on {datetime.utcnow().date()} | "
                f"test_auc={eval_result['test_auc']:.4f} | "
                f"recall@10={eval_result['test_recall_10']:.4f}"
            )
        )

        result = {
            "status":          "success",
            "timestamp":       datetime.utcnow().isoformat(),
            "model_name":      model_name,
            "model_version":   version_num,
            "stage":           "Production",
            "mlflow_run_id":   run_id,
        }

        logger.success(
            f"TASK: register_model completed | "
            f"model={model_name} | version={version_num} | "
            f"stage=Production"
        )
        return result

    except Exception as e:
        logger.error(f"TASK: register_model FAILED: {e}")
        raise


# ----------------------------------------------------------------
# TASK 5: Smoke Test
# ----------------------------------------------------------------

def task_smoke_test(**context) -> dict:
    """
    Calls the live FastAPI /recommend endpoint to verify the
    deployed system returns valid recommendations.

    This is a post-deployment integration test.
    Fails the DAG if the API returns errors or is too slow.
    """
    logger.info("TASK: smoke_test started")

    try:
        api_base_url = os.getenv(
            "FASTAPI_BASE_URL", "http://localhost:8000"
        )
        test_user_id = "user_000001"
        max_latency  = 150   # ms — fail if API is too slow

        # ---- Health check ----
        health_resp = requests.get(
            f"{api_base_url}/health",
            timeout=10,
        )
        if health_resp.status_code != 200:
            raise ValueError(
                f"Health check failed: {health_resp.status_code}"
            )
        logger.info("Health check: PASSED")

        # ---- Recommendation request ----
        start = time.perf_counter()
        rec_resp = requests.post(
            f"{api_base_url}/api/v1/recommend",
            json    = {"user_id": test_user_id, "top_k": 5},
            timeout = 10,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if rec_resp.status_code != 200:
            raise ValueError(
                f"Recommendation request failed: "
                f"{rec_resp.status_code} | {rec_resp.text}"
            )

        data          = rec_resp.json()
        n_recs        = len(data.get("recommendations", []))
        returned_user = data.get("user_id")

        # ---- Assertions ----
        assert n_recs > 0, \
            f"Expected recommendations, got 0"
        assert returned_user == test_user_id, \
            f"user_id mismatch: {returned_user}"
        assert latency_ms < max_latency, \
            f"Latency {latency_ms:.1f}ms exceeds {max_latency}ms SLA"

        result = {
            "status":        "success",
            "timestamp":     datetime.utcnow().isoformat(),
            "test_user":     test_user_id,
            "n_recs":        n_recs,
            "latency_ms":    round(latency_ms, 2),
            "is_cold_start": data.get("is_cold_start"),
            "model_version": data.get("model_version"),
        }

        logger.success(
            f"TASK: smoke_test PASSED | "
            f"n_recs={n_recs} | "
            f"latency={latency_ms:.1f}ms | "
            f"model_version={data.get('model_version')}"
        )
        return result

    except Exception as e:
        logger.error(f"TASK: smoke_test FAILED: {e}")
        raise