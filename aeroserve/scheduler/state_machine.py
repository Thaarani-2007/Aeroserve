import enum
import time
from typing import Optional
from redis.asyncio import Redis


class WorkerState(str, enum.Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


class CircuitBreakerConfig:
    def __init__(
        self,
        heartbeat_ttl_sec: int = 5,
        consecutive_success_threshold: int = 3,
        failure_threshold: int = 2,
    ):
        self.heartbeat_ttl_sec = heartbeat_ttl_sec
        self.consecutive_success_threshold = consecutive_success_threshold
        self.failure_threshold = failure_threshold


class WorkerCircuitBreaker:
    """Manages the lifecycle and state transitions of individual workers in Redis."""

    def __init__(self, redis: Redis, config: Optional[CircuitBreakerConfig] = None):
        self.redis = redis
        self.config = config or CircuitBreakerConfig()

    def _state_key(self, worker_id: str) -> str:
        return f"worker:{worker_id}:state"

    def _heartbeat_key(self, worker_id: str) -> str:
        return f"worker:heartbeat:{worker_id}"

    def _success_counter_key(self, worker_id: str) -> str:
        return f"worker:{worker_id}:consecutive_success"

    def _failure_counter_key(self, worker_id: str) -> str:
        return f"worker:{worker_id}:consecutive_failures"

    async def get_state(self, worker_id: str) -> WorkerState:
        raw = await self.redis.get(self._state_key(worker_id))
        if raw is None:
            return WorkerState.HEALTHY
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return WorkerState(decoded)

    async def set_state(self, worker_id: str, new_state: WorkerState) -> None:
        await self.redis.set(self._state_key(worker_id), new_state.value)
        if new_state == WorkerState.HEALTHY:
            await self.redis.sadd("active_workers:healthy", worker_id)
            await self.redis.srem("active_workers:unhealthy", worker_id)
            await self.redis.srem("active_workers:draining", worker_id)
        elif new_state == WorkerState.UNHEALTHY:
            await self.redis.srem("active_workers:healthy", worker_id)
            await self.redis.sadd("active_workers:unhealthy", worker_id)
            await self.redis.srem("active_workers:draining", worker_id)
        elif new_state == WorkerState.DRAINING:
            await self.redis.srem("active_workers:healthy", worker_id)
            await self.redis.srem("active_workers:unhealthy", worker_id)
            await self.redis.sadd("active_workers:draining", worker_id)

    async def register_worker(self, worker_id: str) -> None:
        await self.redis.sadd("registered_workers", worker_id)
        await self.set_state(worker_id, WorkerState.HEALTHY)

    async def record_heartbeat_failure(self, worker_id: str) -> WorkerState:
        failures = await self.redis.incr(self._failure_counter_key(worker_id))
        await self.redis.delete(self._success_counter_key(worker_id))

        current_state = await self.get_state(worker_id)
        if failures >= self.config.failure_threshold and current_state != WorkerState.UNHEALTHY:
            await self.set_state(worker_id, WorkerState.UNHEALTHY)
            return WorkerState.UNHEALTHY
        return current_state

    async def record_heartbeat_success(self, worker_id: str) -> WorkerState:
        await self.redis.delete(self._failure_counter_key(worker_id))
        current_state = await self.get_state(worker_id)

        if current_state == WorkerState.UNHEALTHY:
            successes = await self.redis.incr(self._success_counter_key(worker_id))
            if successes >= self.config.consecutive_success_threshold:
                await self.set_state(worker_id, WorkerState.HEALTHY)
                await self.redis.delete(self._success_counter_key(worker_id))
                return WorkerState.HEALTHY
            else:
                await self.set_state(worker_id, WorkerState.DRAINING)
                return WorkerState.DRAINING

        elif current_state == WorkerState.DRAINING:
            successes = await self.redis.incr(self._success_counter_key(worker_id))
            if successes >= self.config.consecutive_success_threshold:
                await self.set_state(worker_id, WorkerState.HEALTHY)
                await self.redis.delete(self._success_counter_key(worker_id))
                return WorkerState.HEALTHY
            return WorkerState.DRAINING

        return WorkerState.HEALTHY