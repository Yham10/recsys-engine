-- ================================================================
-- PostgreSQL Initialization Script
-- Runs ONCE on first container start (fresh volume)
-- Creates separate databases for each service
-- ================================================================

-- feast_db: Feast offline store (feature tables)
CREATE DATABASE feast_db;

-- mlflow_db: MLflow experiment tracking metadata
CREATE DATABASE mlflow_db;

-- airflow_db: Airflow pipeline metadata
CREATE DATABASE airflow_db;

-- Grant full access to our user
GRANT ALL PRIVILEGES ON DATABASE feast_db   TO recsys_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db  TO recsys_user;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO recsys_user;

-- Connect to feast_db and create the feast schema
\c feast_db
CREATE SCHEMA IF NOT EXISTS feast;
GRANT ALL ON SCHEMA feast TO recsys_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA feast
    GRANT ALL ON TABLES TO recsys_user;

-- Connect back to default db
\c recsys_db
DO $$
BEGIN
    RAISE NOTICE 'All databases created successfully at %', NOW();
END $$;