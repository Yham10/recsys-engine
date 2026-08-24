"""
Orchestration Package
=====================
Apache Airflow DAGs for automating the RecSys ML pipeline.

DAGs:
    daily_training_pipeline          → Full retrain cycle (daily)
    feature_materialization_pipeline → Feature refresh (every 6h)
    data_quality_pipeline            → Data drift monitoring (daily)
"""