FROM python:3.11-slim

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY autoscaler/requirements.txt /app/requirements.txt
RUN pip install --default-timeout=1000 -r requirements.txt
COPY config.py /app/config.py
COPY autoscaler/ /app/autoscaler/

ENV PYTHONPATH=/app

CMD ["python", "-u", "autoscaler/main.py"]