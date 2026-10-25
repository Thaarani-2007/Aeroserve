FROM python:3.11-slim

WORKDIR /app

COPY gateway/requirements.txt ./requirements.txt
RUN pip install --default-timeout=1000 -r requirements.txt
COPY gateway/ ./gateway/
COPY config.py ./config.py

ENV PYTHONPATH=/app

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--backlog", "2048"]
