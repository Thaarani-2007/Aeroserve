FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for psutil and ML frameworks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt .
RUN pip install --default-timeout=3000 --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --default-timeout=3000 --no-cache-dir -r requirements.txt
# Copy the necessary files
COPY worker/ ./worker/
COPY config.py .

ENV PYTHONPATH=/app
CMD ["python", "-u", "worker/main.py"]