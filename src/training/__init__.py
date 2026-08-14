"""
Training Package
================
Implements the Two-Tower Neural Network for e-commerce recommendation.

Components:
    dataset.py      →  PyTorch Dataset — loads Parquet, serves batches
    model.py        →  Two-Tower architecture (user tower + item tower)
    metrics.py      →  Recall@K, NDCG@K, AUC computation
    trainer.py      →  Full training loop with MLflow tracking
    run_training.py →  CLI entry point called by Airflow DAG
"""