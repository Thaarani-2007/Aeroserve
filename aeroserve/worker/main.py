import asyncio
import json
import logging
import signal
import time
import uuid
import os
import sys
import importlib.util
from collections import deque, defaultdict
from enum import Enum  

sys.path.insert(0, "/app")

from config import settings
from worker.heartbeat import HeartbeatEmitter
from prometheus_client import start_http_server, Gauge, Counter, Histogram

WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"

# --- PROMETHEUS METRICS ---
CB_STATE = Gauge("aeroserve_circuit_state", "Circuit Breaker State", ["worker_id"])
INFERENCE_TIME = Histogram("aeroserve_inference_time_ms", "Time taken for inference in ms", ["worker_id", "model_name"])
REQUEUED_TASKS = Counter("aeroserve_requeued_tasks_total", "Number of tasks requeued", ["worker_id"])
AEROSERVE_BATCH_SIZE = Histogram("aeroserve_batch_size", "Number of items in a batch", buckets=(1, 2, 4, 8, 16, 32, 64))
AEROSERVE_BATCH_TRIGGER = Counter("aeroserve_batch_trigger_total", "Batch trigger reason", ["worker_id", "reason"])
AEROSERVE_BATCH_WAIT_TIME = Histogram("aeroserve_batch_wait_time_seconds", "Time first request waited in buffer", buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0))

CB_STATE.labels(worker_id=WORKER_ID).set(0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [AeroServe.Worker]: %(message)s")
logger = logging.getLogger("AeroServe.Worker")

class CircuitState(Enum):
    CLOSED = "CLOSED"       
    OPEN = "OPEN"           
    HALF_OPEN = "HALF_OPEN" 

class CircuitBreaker:
    def __init__(self, failure_threshold: int = settings.CB_FAILURE_THRESHOLD, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def set_state(self, new_state: CircuitState):
        self.state = new_state
        state_map = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}
        CB_STATE.labels(worker_id=WORKER_ID).set(state_map[new_state])

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED: return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.warning("Circuit Breaker entering HALF_OPEN state.")
                self.set_state(CircuitState.HALF_OPEN)
                return True
            return False
        return True

    def record_success(self):
        if self.state != CircuitState.CLOSED:
            logger.info("Worker recovered completely.")
        self.set_state(CircuitState.CLOSED)
        self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.set_state(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit Breaker TRIPPED to OPEN after {self.failure_count} crashes!")
            self.set_state(CircuitState.OPEN)

class DynamicModelRegistry:
    def __init__(self, models_dir="/app/models"):
        self.models_dir = models_dir
        self.processors = {}
        self.configs = {} 
        self.load_all_processors()

    def load_model(self, model_name: str) -> bool:
        """FIX: Dynamically load a specific model on-demand (Zero-Touch support)"""
        folder_path = os.path.join(self.models_dir, model_name)
        config_path = os.path.join(folder_path, "model_config.json")
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.configs[model_name] = config 
            
            proc_info = config.get("processor")
            if proc_info:
                file_path = os.path.join(folder_path, proc_info["file"])
                class_name = proc_info["class_name"]
                spec = importlib.util.spec_from_file_location(class_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ProcessorClass = getattr(module, class_name)
                self.processors[model_name] = ProcessorClass()
                logger.info(f"Zero-Touch: Dynamically loaded model {model_name} into VRAM.")
                return True
        except Exception as e:
            logger.error(f"Failed to dynamically load {model_name}: {e}")
        return False

    def load_all_processors(self):
        if not os.path.exists(self.models_dir): return
        for folder in os.listdir(self.models_dir):
            if not folder.startswith(".tmp_"):
                self.load_model(folder)

    def run_inference(self, requests: list) -> list[dict]:
        if not requests: return []
        model_name = requests[0].model_name
        
        # FIX: On-demand loading
        if model_name not in self.processors:
            success = self.load_model(model_name)
            if not success:
                return [{"status": "error", "error": f"Model {model_name} not found"} for _ in requests]
                
        processor = self.processors[model_name]
        has_batch_support = getattr(processor, "supports_batching", False)
        
        # FIX: True vectorized execution
        try:
            if has_batch_support:
                payloads = [req.payload for req in requests]
                return processor.execute_batch(payloads)
            else:
                # Fallback for models that explicitly don't support batching
                results = []
                for req in requests:
                    results.append(processor.execute(req.payload))
                return results
        except Exception as e:
            logger.error(f"Inference crash on {model_name}: {e}")
            raise

class InferenceRequest:
    def __init__(self, raw_data: str, parsed_data: dict, registry: DynamicModelRegistry):
        self.request_id = parsed_data["request_id"]
        self.model_name = parsed_data.get("model_name", "dummy_model")
        self.priority = parsed_data["priority"]
        self.created_at = parsed_data["created_at"]
        self.payload = parsed_data.get("payload")
        self.raw_data = raw_data  
        
        model_config = registry.configs.get(self.model_name, {})
        slas = model_config.get("slas", {}).get("priority_timeouts_s", {})
        sla_timeout = slas.get(self.priority, 1.0) 
        self.deadline = self.created_at + sla_timeout

class Batch:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.requests: list[InferenceRequest] = []
        self.flush_reason = None
        self.batch_start_time = time.time()  # FIX: Track actual batch formation time

class ReliableTaskFetcher:
    def __init__(self, redis, worker_id: str, registry: DynamicModelRegistry):
        self.redis = redis
        self.worker_id = worker_id
        self.processing_queue = f"queue:processing:{self.worker_id}"
        self.registry = registry

    async def fetch_task(self, source_queue: str) -> InferenceRequest | None:
        try:
            raw_payload = await self.redis.lmove(source_queue, self.processing_queue, "RIGHT", "LEFT")
            if not raw_payload: return None
            payload_str = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else raw_payload
            return InferenceRequest(raw_data=payload_str, parsed_data=json.loads(payload_str), registry=self.registry)
        except Exception as e:
            return None

    async def complete_task(self, raw_data_str: str) -> None:
        await self.redis.lrem(self.processing_queue, 1, raw_data_str)

    async def requeue_task(self, req: InferenceRequest) -> None:
        try:
            await self.redis.lpush(f"queue:{req.priority}", req.raw_data)
            await self.complete_task(req.raw_data)
            REQUEUED_TASKS.labels(worker_id=self.worker_id).inc() 
        except Exception:
            pass

class WRRScheduler:
    def __init__(self, redis_client, fetcher: ReliableTaskFetcher):
        self.redis = redis_client
        self.fetcher = fetcher

    async def calculate_effective_weights(self) -> dict[str, int]:
        weights = settings.BASE_WEIGHTS.copy()
        current_time = time.time()
        pipe = self.redis.pipeline()
        pipe.lrange("queue:standard", -1, -1)
        pipe.lrange("queue:background", -1, -1)
        std_oldest, bg_oldest = await pipe.execute()
        
        if std_oldest:
            if current_time - json.loads(std_oldest[0])["created_at"] > 0.1:
                weights["standard"] = 5
        if bg_oldest:
            wait_time = current_time - json.loads(bg_oldest[0])["created_at"]
            if wait_time > 30.0: weights["background"] = 3
            if wait_time > 60.0: weights["background"] = 5
        return weights

    async def fetch_wrr(self) -> list[InferenceRequest]:
        weights = await self.calculate_effective_weights()
        fetched = []
        for queue_name, count in weights.items():
            for _ in range(count):
                task = await self.fetcher.fetch_task(f"queue:{queue_name}")
                if task: fetched.append(task)
                else: break 
        return fetched

class BatchBuilder:
    def __init__(self, scheduler: WRRScheduler, registry: DynamicModelRegistry):
        self.scheduler = scheduler
        self.registry = registry
        # FIX: Bucket requests by model to prevent Head-of-Line blocking
        self.model_buffers = defaultdict(list)
        self.active_batches = {}

    async def build_next_batch(self, shutdown_event: asyncio.Event) -> Batch | None:
        while not shutdown_event.is_set():
            # 1. Fetch new tasks and bucket them by model
            fetched = await self.scheduler.fetch_wrr()
            for req in fetched:
                self.model_buffers[req.model_name].append(req)
                if req.model_name not in self.active_batches:
                    self.active_batches[req.model_name] = Batch(req.model_name)

            now = time.time()

            # 2. Check all active batches for flush conditions
            for model_name, batch in list(self.active_batches.items()):
                buffer = self.model_buffers[model_name]
                
                model_config = self.registry.configs.get(model_name, {})
                max_batch_size = model_config.get("batching", {}).get("max_batch_size", settings.DEFAULT_MAX_BATCH_SIZE)
                batch_timeout_ms = model_config.get("batching", {}).get("batch_timeout_ms", settings.DEFAULT_BATCH_TIMEOUT_MS)

                # Move items from buffer to batch
                while buffer and len(batch.requests) < max_batch_size:
                    batch.requests.append(buffer.pop(0))
                
                if not batch.requests:
                    del self.active_batches[model_name]
                    continue

                # Trigger 1: Max Size
                if len(batch.requests) >= max_batch_size:
                    batch.flush_reason = "MAX_SIZE"
                    del self.active_batches[model_name]
                    return batch

                # Trigger 2: SLA Deadline Danger
                earliest_deadline = min(req.deadline for req in batch.requests)
                if (earliest_deadline - now) <= 0.05: # Give 50ms buffer to execute
                    batch.flush_reason = "EARLIEST_DEADLINE"
                    del self.active_batches[model_name]
                    return batch
                
                # Trigger 3: Batch Timeout (FIXED: Measured from batch creation, not ingestion)
                if (now - batch.batch_start_time) * 1000 >= batch_timeout_ms:
                    batch.flush_reason = "BATCH_TIMEOUT"
                    del self.active_batches[model_name]
                    return batch

            try:
                await asyncio.sleep(0.01) 
            except asyncio.CancelledError:
                break
                
        # Flush everything on shutdown
        for model_name, batch in self.active_batches.items():
            if batch.requests:
                batch.flush_reason = "SHUTDOWN"
                return batch
        return None

class InferenceWorker:
    def __init__(self, processor_registry, redis_client, fetcher: ReliableTaskFetcher):
        self.processor_registry = processor_registry
        self.redis = redis_client
        self.fetcher = fetcher
        self.circuit_breaker = CircuitBreaker() 

    def log_metrics(self, batch: Batch, inference_ms: float):
        counts = {"premium": 0, "standard": 0, "background": 0}
        wait_time_s = 0.0
        
        for req in batch.requests:
            counts[req.priority] += 1
            
        AEROSERVE_BATCH_SIZE.observe(len(batch.requests))
        if batch.flush_reason:
            AEROSERVE_BATCH_TRIGGER.labels(worker_id=WORKER_ID, reason=batch.flush_reason.lower()).inc()
            
        if batch.requests:
            wait_time_s = time.time() - min(req.created_at for req in batch.requests)
            AEROSERVE_BATCH_WAIT_TIME.observe(wait_time_s)

        INFERENCE_TIME.labels(worker_id=WORKER_ID, model_name=batch.model_name).observe(inference_ms)
        logger.info(f"Batch flushed [{batch.model_name}] | Reason: {batch.flush_reason} | Size: {len(batch.requests)} | P:{counts['premium']} S:{counts['standard']} B:{counts['background']} | Inference: {inference_ms:.1f}ms")

    async def execute(self, batch: Batch):
        if not batch.requests: return
            
        if not self.circuit_breaker.can_execute():
            logger.warning("Worker unhealthy. Requeueing batch.")
            for req in batch.requests: await self.fetcher.requeue_task(req)
            return
            
        start_time = time.monotonic()
        
        try:
            results = await asyncio.to_thread(self.processor_registry.run_inference, batch.requests)
            
            has_success = any(res.get("status") != "error" for res in results) if results else False
            if has_success: self.circuit_breaker.record_success()
            else: self.circuit_breaker.record_failure()
            
            for req, result_data in zip(batch.requests, results):
                response_payload = {"request_id": req.request_id, "status": "success" if result_data.get("status") != "error" else "error", "data": result_data}
                await self.redis.publish(f"response:{req.request_id}", json.dumps(response_payload))
                await self.fetcher.complete_task(req.raw_data)
                
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Fatal inference crash: {e}")
            for req in batch.requests:
                error_payload = {"request_id": req.request_id, "status": "error", "error": str(e)}
                await self.redis.publish(f"response:{req.request_id}", json.dumps(error_payload))
                await self.fetcher.complete_task(req.raw_data) # Remove from processing queue since we answered the client
        finally:
            self.log_metrics(batch, (time.monotonic() - start_time) * 1000)

async def main():
    start_http_server(8002)
    
    import redis.asyncio as redis_async
    # FIX: Use REDIS_URL mapping from Phase 2
    redis_client = redis_async.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    heartbeat = HeartbeatEmitter(redis_client, worker_id=WORKER_ID)
    await heartbeat.start()
    
    registry = DynamicModelRegistry()
    task_fetcher = ReliableTaskFetcher(redis_client, worker_id=WORKER_ID, registry=registry)
    scheduler = WRRScheduler(redis_client, task_fetcher)
    batch_builder = BatchBuilder(scheduler, registry=registry)
    worker = InferenceWorker(registry, redis_client, task_fetcher)
    
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.warning("Shutdown signal received. Finishing active batches...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info(f"Worker {WORKER_ID} started. Listening for requests...")
    
    while not shutdown_event.is_set():
        batch = await batch_builder.build_next_batch(shutdown_event)
        if batch and batch.requests:
            await worker.execute(batch)
            
    await heartbeat.stop()
    await redis_client.aclose()
    logger.info("Worker shut down cleanly.")

if __name__ == "__main__":
    asyncio.run(main())