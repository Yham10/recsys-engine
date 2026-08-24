# ================================================================
# Airflow Dockerfile
# ================================================================
FROM apache/airflow:2.8.1-python3.11

ARG AIRFLOW_VERSION=2.8.1

LABEL maintainer="recsys-team"
LABEL description="Airflow with RecSys ML dependencies"

USER root

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt

RUN pip install \
      --no-cache-dir \
      --timeout=120 \
      --upgrade-strategy only-if-needed \
      "apache-airflow==${AIRFLOW_VERSION}" \
      -r /tmp/requirements-airflow.txt \
    && python -c "import airflow; assert airflow.__version__ == '${AIRFLOW_VERSION}', airflow.__version__; print('Airflow version:', airflow.__version__)" \
    && pip check

# ── FIX: force email-validator>=2.0 AFTER main install ──
# pydantic (via feast→fastapi) needs email-validator>=2.0 at runtime,
# but flask-appbuilder pins <2.0 in its metadata. Installing with
# --no-deps bypasses pip's resolver — at runtime both work fine with 2.x.
RUN pip install --no-cache-dir --no-deps "email-validator>=2.0.0"