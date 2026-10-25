import asyncio
import logging
from redis.asyncio import Redis

logger = logging.getLogger("AeroServe.Worker.Heartbeat")

class HeartbeatEmitter:
    """Runs as a lightweight background coroutine inside the worker container."""

    def __init__(self, redis: Redis, worker_id: str, heartbeat_interval_sec: float = 2.0, ttl_sec: int = 5):
        self.redis = redis
        self.worker_id = worker_id
        self.interval = heartbeat_interval_sec
        self.ttl = ttl_sec
        self.is_running = False
        self._task: asyncio.Task = None

    async def start(self):
        self.is_running = True
        await self.redis.sadd("registered_workers", self.worker_id)
        self._task = asyncio.create_task(self._pulse_loop())

    async def _pulse_loop(self):
        heartbeat_key = f"worker:heartbeat:{self.worker_id}"
        while self.is_running:
            try:
                await self.redis.set(heartbeat_key, "1", ex=self.ttl)
            except Exception as e:
                logger.error(f"Failed to publish heartbeat: {e}")
            await asyncio.sleep(self.interval)

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # FIX: Clean up registry on graceful shutdown
        try:
            await self.redis.srem("registered_workers", self.worker_id)
            await self.redis.delete(f"worker:heartbeat:{self.worker_id}")
        except Exception:
            pass