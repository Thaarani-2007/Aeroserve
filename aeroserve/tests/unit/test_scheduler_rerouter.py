import json
import pytest
from scheduler.rerouter import RequestRerouter

@pytest.mark.asyncio
async def test_recover_orphaned_requests(redis):
    rerouter = RequestRerouter(redis)
    dead_worker_id = "worker-dead-1"
    
    # Simulate tasks trapped in a dead worker's processing queue
    tasks = [
        {"request_id": "req-1", "priority": "premium"},
        {"request_id": "req-2", "priority": "standard"}
    ]
    for task in tasks:
        await redis.lpush(f"queue:processing:{dead_worker_id}", json.dumps(task))
        
    recovered_count = await rerouter.recover_orphaned_requests(dead_worker_id)
    
    assert recovered_count == 2
    assert await redis.llen(f"queue:processing:{dead_worker_id}") == 0
    assert await redis.llen("queue:premium") == 1
    assert await redis.llen("queue:standard") == 1
    
    # Verify LPUSH maintained priority at the front of the queue
    premium_task = json.loads(await redis.rpop("queue:premium"))
    assert premium_task["requeued"] is True