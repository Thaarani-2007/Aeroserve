import asyncio
import json
import time
import logging
from prometheus_client import start_http_server, Gauge
import redis.asyncio as redis_async

from config import settings
from scheduler.state_machine import WorkerCircuitBreaker, CircuitBreakerConfig, WorkerState
from scheduler.rerouter import RequestRerouter

logger = logging.getLogger("AeroServe.Scheduler")

QUEUE_DEPTH = Gauge('aeroserve_queue_depth', 'Current requests in queue', ['priority'])
OLDEST_AGE = Gauge('aeroserve_oldest_request_age_seconds', 'Age of oldest request', ['priority'])
AEROSERVE_TARGET_WORKERS = Gauge("aeroserve_target_workers", "Number of worker replicas requested by the autoscaler")

async def metrics_publisher_loop(redis_client: redis_async.Redis):
    queues = ["premium", "standard", "background"]
    logger.info("Metrics publisher loop started on port 8001")
    
    while True:
        try:
            for q_name in queues:
                queue_key = f"queue:{q_name}"
                depth = await redis_client.llen(queue_key)
                QUEUE_DEPTH.labels(priority=q_name).set(depth)
                
                if depth > 0:
                    oldest_raw = await redis_client.lindex(queue_key, -1)
                    if oldest_raw:
                        task = json.loads(oldest_raw)
                        age = time.time() - task["created_at"]
                        OLDEST_AGE.labels(priority=q_name).set(age)
                else:
                    OLDEST_AGE.labels(priority=q_name).set(0)
                    
        except Exception as e:
            logger.error(f"Error publishing metrics: {e}")
            
        await asyncio.sleep(2)

async def health_check_loop(redis_client: redis_async.Redis, cb: WorkerCircuitBreaker, rerouter: RequestRerouter):
    """FIX: The missing engine. Actively scans heartbeats and triggers failover."""
    logger.info("Worker Health Monitor started.")
    
    while True:
        try:
            # Get all workers that registered themselves
            workers = await redis_client.smembers("registered_workers")
            
            for w_id in workers:
                is_alive = await redis_client.exists(f"worker:heartbeat:{w_id}")
                
                if not is_alive:
                    # Heartbeat missing! Record failure in the circuit breaker
                    new_state = await cb.record_heartbeat_failure(w_id)
                    
                    if new_state == WorkerState.UNHEALTHY:
                        logger.warning(f"Worker {w_id} isolated. Rerouting orphaned tasks...")
                        await rerouter.recover_orphaned_requests(w_id)
                        
                        # Remove from active registry so we don't endlessly reroute
                        await redis_client.srem("registered_workers", w_id)
                else:
                    # Heartbeat active, record success
                    await cb.record_heartbeat_success(w_id)
                    
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            
        await asyncio.sleep(settings.HEARTBEAT_INTERVAL_SEC)

async def main():
    start_http_server(8001)
    redis_client = redis_async.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    config = CircuitBreakerConfig(
        failure_threshold=settings.CB_FAILURE_THRESHOLD,
        consecutive_success_threshold=settings.CB_CONSECUTIVE_SUCCESS_THRESHOLD
    )
    cb = WorkerCircuitBreaker(redis_client, config)
    rerouter = RequestRerouter(redis_client)
    
    metrics_task = asyncio.create_task(metrics_publisher_loop(redis_client))
    health_task = asyncio.create_task(health_check_loop(redis_client, cb, rerouter))
    
    logger.info("Scheduler started successfully.")
    
    await asyncio.gather(metrics_task, health_task)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())