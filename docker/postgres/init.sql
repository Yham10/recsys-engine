-- ================================================================
-- PostgreSQL Initialization Script
-- Creates separate schemas for MLflow and Feast
-- Runs automatically on first container start
-- ================================================================

-- Create dedicated database schemas
CREATE SCHEMA IF NOT EXISTS mlflow;
CREATE SCHEMA IF NOT EXISTS feast;

-- Create a read-only user for analytics/debugging
CREATE USER recsys_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE recsys_db TO recsys_readonly;
GRANT USAGE ON SCHEMA mlflow TO recsys_readonly;
GRANT USAGE ON SCHEMA feast TO recsys_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA mlflow TO recsys_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA feast TO recsys_readonly;

-- Ensure future tables are also accessible to read-only user
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow
    GRANT SELECT ON TABLES TO recsys_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA feast
    GRANT SELECT ON TABLES TO recsys_readonly;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully at %', NOW();
END $$;