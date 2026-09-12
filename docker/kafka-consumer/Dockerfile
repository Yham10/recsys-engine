FROM python:3.11-slim
WORKDIR /app

# Install only consumer dependencies with extended timeout
COPY requirements-kafka-consumer.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-kafka-consumer.txt

# Copy only the consumer script
COPY src/data_generator/kafka_consumer.py /app/src/data_generator/kafka_consumer.py
ENV PYTHONPATH=/app/src

CMD ["python", "src/data_generator/kafka_consumer.py"]