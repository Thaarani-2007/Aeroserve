import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status, BackgroundTasks
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

from config import settings
from gateway.schemas import HealthResponse, PredictRequest, PredictResponse

# --- Prometheus Metrics ---
AEROSERVE_QUEUE_DEPTH = Gauge("aeroserve_queue_depth", "Current depth of the priority queues", ["priority"])
AEROSERVE_REQUEST_COUNT = Counter("aeroserve_requests_total", "Total inference requests received", ["priority", "model_name"])
AEROSERVE_OLDEST_AGE = Gauge("aeroserve_oldest_request_age_seconds", "Age of the oldest request", ["priority"])

AEROSERVE_LATENCY = Histogram(
    "aeroserve_request_latency_seconds", 
    "End-to-end latency", 
    ["priority"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0)
)
AEROSERVE_INFERENCE_STATUS = Counter("aeroserve_inference_status_total", "Outcome of inference requests", ["priority", "status"])
AEROSERVE_IN_FLIGHT = Gauge("aeroserve_in_flight_requests", "Requests currently being processed", ["priority"])
AEROSERVE_DB_WRITES = Counter("aeroserve_db_writes_total", "Postgres write success/fail", ["status"])

KNOWN_PRIORITIES = ["premium", "standard", "background"]

# --- DYNAMIC MODEL REGISTRY ---
ENABLE_CHAOS_MODE = os.getenv("ENABLE_CHAOS_MODE", "True").lower() == "true"
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")  # Changed to /app/models to match Docker
model_registry = {}

def load_model_registry():
    """Reloads the models in memory so zero-touch updates are recognized"""
    if not os.path.exists(MODELS_DIR):
        print(f"[Gateway] Warning: Models directory '{MODELS_DIR}' not found.")
        return
    
    current_models = {}
    for model_folder in os.listdir(MODELS_DIR):
        # Skip temporary folders created by the CLI during atomic swap
        if model_folder.startswith(".tmp_"):
            continue
            
        config_path = os.path.join(MODELS_DIR, model_folder, "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    model_name = config["model_name"]
                    current_models[model_name] = config
                    print(f"[Gateway] Registered Model: {model_name}")
            except Exception as e:
                print(f"[Gateway] Failed to load config for {model_folder}: {e}")
                
    # Update global registry atomically
    global model_registry
    model_registry = current_models

async def update_prometheus_metrics(redis_client: redis.Redis):
    while True:
        try:
            # Periodically reload models to support the CLI zero-touch deployment
            load_model_registry()
            
            for priority in KNOWN_PRIORITIES:
                queue_name = f"queue:{priority}"
                depth = await redis_client.llen(queue_name)
                AEROSERVE_QUEUE_DEPTH.labels(priority=priority).set(depth)
                
                if depth > 0:
                    oldest_req_bytes = await redis_client.lrange(queue_name, 0, 0)
                    if oldest_req_bytes:
                        req_data = json.loads(oldest_req_bytes[0])
                        created_at = req_data.get("created_at", time.time())
                        AEROSERVE_OLDEST_AGE.labels(priority=priority).set(time.time() - created_at)
                else:
                    AEROSERVE_OLDEST_AGE.labels(priority=priority).set(0.0)
        except Exception as e:
            print(f"[Gateway] Metrics/Registry error: {e}")
        await asyncio.sleep(2)

# --- POSTGRES LOGGING ---
async def log_request_to_db(pool, req_id: str, model: str, priority: str, req_status: str, result: dict | None, latency: float):
    # Upsert query handles both the initial "pending" log and the final "success/error" update
    query = """
        INSERT INTO inference_logs (request_id, model_name, priority, status, inference_result, latency_ms)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (request_id) 
        DO UPDATE SET 
            status = EXCLUDED.status, 
            inference_result = EXCLUDED.inference_result, 
            latency_ms = EXCLUDED.latency_ms
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(query, req_id, model, priority, req_status, json.dumps(result) if result else None, latency)
        AEROSERVE_DB_WRITES.labels(status="success").inc()
    except Exception as e:
        AEROSERVE_DB_WRITES.labels(status="error").inc()
        print(f"[Gateway] DB Logging Error for {req_id}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    load_model_registry()
    
    app.state.redis_pool = redis.ConnectionPool.from_url(settings.REDIS_URL, max_connections=500, decode_responses=True)
    app.state.redis = redis.Redis(connection_pool=app.state.redis_pool)
    
    app.state.db_pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=5, max_size=20)
    try:
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    request_id UUID PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    priority VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    inference_result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    latency_ms REAL
                );
            """)
        print("[Gateway] PostgreSQL inference_logs table is ready.")
    except Exception as e:
        print(f"[Gateway] Table creation safely skipped: {e}")
        
    metrics_task = asyncio.create_task(update_prometheus_metrics(app.state.redis))
    yield
    
    metrics_task.cancel()
    await app.state.redis.aclose()
    await app.state.redis_pool.disconnect()
    await app.state.db_pool.close()

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

def get_redis_client(request: Request) -> redis.Redis:
    return request.app.state.redis

@app.post("/chaos/crash", tags=["Chaos Engineering"], status_code=status.HTTP_202_ACCEPTED)
async def trigger_chaos(queue_name: str = "standard", model: str = "resnet50_classifier", redis_client: redis.Redis = Depends(get_redis_client)):
    if not ENABLE_CHAOS_MODE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chaos Engineering mode disabled.")
    
    request_id = f"chaos-{uuid.uuid4().hex[:8]}"
    envelope = {
        "request_id": request_id,
        "model_name": model,
        "created_at": time.time(),
        "priority": queue_name,
        "payload": {"image_data": "POISON_PILL"} 
    }
    try:
        await redis_client.rpush(f"queue:{queue_name}", json.dumps(envelope))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Chaos injection failed: {exc}")
    return {"status": "Poison pill injected", "request_id": request_id}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(redis_client: redis.Redis = Depends(get_redis_client)):
    try:
        redis_ok = await redis_client.ping()
    except Exception:
        redis_ok = False
    
    # FIX: Expose loaded models so the CLI deployment verification works
    loaded_models = list(model_registry.keys())
    return HealthResponse(
        status="healthy" if redis_ok else "degraded", 
        redis_connected=redis_ok,
        models_loaded=loaded_models
    )

@app.post("/predict", response_model=PredictResponse, tags=["Inference Routing"])
async def predict(
    request_data: PredictRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    model_name = request_data.model_name
    
    # FIX: Prevent phantom requests. Act as a strict bouncer.
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or not loaded.")

    priority_val = getattr(request_data.priority, 'value', request_data.priority)
    queue_name = f"queue:{priority_val}"
    
    slas = model_registry[model_name].get("slas", {})
    dynamic_timeout = slas.get("priority_timeouts_s", {}).get(priority_val, slas.get("max_queue_wait_s", 45.0))

    request_id = str(uuid.uuid4())
    timestamp = time.time()
    
    envelope = {
        "request_id": request_id,
        "model_name": model_name,
        "created_at": timestamp,                    
        "priority": priority_val,
        "user_tier": getattr(request_data.user_tier, 'value', request_data.user_tier),
        "payload": request_data.payload.model_dump() # Pass the dict, not the Pydantic object
    }
    
    pubsub = redis_client.pubsub()
    response_channel = f"response:{request_id}"
    await pubsub.subscribe(response_channel)
    
    # FIX: Ensure no requests are lost to DB if queue crashes
    background_tasks.add_task(log_request_to_db, request.app.state.db_pool, request_id, model_name, priority_val, "pending", None, 0.0)
    
    AEROSERVE_IN_FLIGHT.labels(priority=priority_val).inc()
    
    try:
        try:
            await redis_client.rpush(queue_name, json.dumps(envelope))
            AEROSERVE_REQUEST_COUNT.labels(priority=priority_val, model_name=model_name).inc()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Queue backend unavailable.")
            
        try:
            async with asyncio.timeout(dynamic_timeout): 
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        result = json.loads(message["data"])
                        latency_s = time.time() - timestamp
                        latency_ms = latency_s * 1000
                        
                        AEROSERVE_LATENCY.labels(priority=priority_val).observe(latency_s)
                        
                        if result.get("status") == "error":
                            AEROSERVE_INFERENCE_STATUS.labels(priority=priority_val, status="worker_error").inc()
                            background_tasks.add_task(log_request_to_db, request.app.state.db_pool, request_id, model_name, priority_val, "error", {"error": result.get("error")}, latency_ms)
                            raise HTTPException(status_code=500, detail=result.get("error"))
                        
                        AEROSERVE_INFERENCE_STATUS.labels(priority=priority_val, status="success").inc()
                        background_tasks.add_task(log_request_to_db, request.app.state.db_pool, request_id, model_name, priority_val, "success", result.get("data"), latency_ms)
                        
                        return PredictResponse(
                            request_id=request_id,
                            status="completed",
                            queue=queue_name,
                            timestamp=timestamp,
                            inference_result=result.get("data")
                        )
        except asyncio.TimeoutError:
            latency_s = time.time() - timestamp
            latency_ms = latency_s * 1000
            AEROSERVE_LATENCY.labels(priority=priority_val).observe(latency_s)
            AEROSERVE_INFERENCE_STATUS.labels(priority=priority_val, status="timeout").inc()
            
            background_tasks.add_task(log_request_to_db, request.app.state.db_pool, request_id, model_name, priority_val, "timeout", None, latency_ms)
            raise HTTPException(status_code=504, detail=f"Worker timeout for SLA: {dynamic_timeout}s")
        finally:
            await pubsub.unsubscribe(response_channel)
            await pubsub.close() 
    finally:
        AEROSERVE_IN_FLIGHT.labels(priority=priority_val).dec()