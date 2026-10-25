import json
import logging
from typing import List, Dict, Any
from redis.asyncio import Redis

logger = logging.getLogger("AeroServe.Scheduler.Rerouter")


class RequestRerouter:
    """Detects requests assigned to dead/unhealthy workers and requeues them safely."""

    def __init__(self, redis: Redis):
        self.redis = redis

    def _processing_queue(self, worker_id: str) -> str:
        return f"queue:processing:{worker_id}"

    async def recover_orphaned_requests(self, dead_worker_id: str) -> int:
        processing_queue = self._processing_queue(dead_worker_id)
        requeued_count = 0

        while True:
            # Atomic fetch from dead worker's processing queue
            raw_payload = await self.redis.rpop(processing_queue)
            if not raw_payload:
                break

            payload_str = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else raw_payload
            try:
                request_data: Dict[str, Any] = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.error(f"Malformed payload dropped during recovery: {payload_str}")
                continue

            priority = request_data.get("priority", "standard").lower()
            target_queue = f"queue:{priority}"

            # Re-enqueue at the front (LPUSH) so orphaned requests retain priority
            request_data["requeued"] = True
            await self.redis.lpush(target_queue, json.dumps(request_data))
            requeued_count += 1

        if requeued_count > 0:
            logger.warning(
                f"Recovered and requeued {requeued_count} orphaned request(s) from worker {dead_worker_id}."
            )
        return requeued_count