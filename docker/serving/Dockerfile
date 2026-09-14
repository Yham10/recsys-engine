FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.3.1"

COPY requirements-serving.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Fix pydantic/fastapi email-validator conflict
RUN pip install --no-cache-dir --no-deps "email-validator>=2.0.0"

# Copy serving code
COPY src/serving/ /app/

# Copy shared modules that serving depends on
COPY src/config_paths.py /app/config_paths.py
COPY src/training/dataset.py /app/dataset.py
COPY src/training/model.py /app/model.py

# Copy Feast feature repo
COPY src/feature_store/feature_repo/ /app/feature_repo/

RUN mkdir -p /app/logs

# Add training dir to Python path so model_loader can find dataset, model, etc.
ENV PYTHONPATH="/app:/app/training"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]