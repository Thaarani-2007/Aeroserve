import json
import pytest
from worker.main import ReliableTaskFetcher

@pytest.mark.asyncio
async def test_reliable_task_fetcher_atomic_move(redis):
    worker_id = "worker-fetch-1"
    fetcher = ReliableTaskFetcher(redis, worker_id)
    
    # Seed a task in the premium queue
    task_payload = {"request_id": "req-1", "priority": "premium", "created_at": 1000}
    await redis.lpush("queue:premium", json.dumps(task_payload))
    
    # Fetch task
    task = await fetcher.fetch_task("queue:premium")
    assert task is not None
    assert task.request_id == "req-1"
    
    # Verify it was removed from premium and added to processing
    assert await redis.llen("queue:premium") == 0
    assert await redis.llen(fetcher.processing_queue) == 1
    
    # Complete task
    await fetcher.complete_task(task.raw_data)
    assert await redis.llen(fetcher.processing_queue) == 0