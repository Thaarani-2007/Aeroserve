FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scheduler/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY scheduler/ ./scheduler/
COPY config.py .

ENV PYTHONPATH=/app
CMD ["python", "-u", "scheduler/main.py"]