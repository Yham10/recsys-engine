# ================================================================
# MLflow Server Dockerfile
# ================================================================
FROM python:3.11-slim

LABEL maintainer="recsys-team"
LABEL description="MLflow tracking server with PostgreSQL and S3/MinIO support"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0 \
    pymysql==1.1.0

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlflow
USER mlflow
WORKDIR /mlflow

EXPOSE 5000