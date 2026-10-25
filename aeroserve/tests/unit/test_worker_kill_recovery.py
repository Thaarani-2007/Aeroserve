import asyncio
import json
import pytest
from scheduler.state_machine import WorkerCircuitBreaker, CircuitBreakerConfig, WorkerState
from scheduler.rerouter import RequestRerouter

@pytest.mark.asyncio
async def test_end_to_end_worker_crash_and_recovery(redis):
    """
    Simulates:
    1. A worker pulling a task.
    2. The worker container crashing (heartbeats stop, task stuck in processing).
    3. The scheduler detecting the crash, isolating the worker, and rerouting the task.
    """
    worker_id = "worker-doomed-container"
    
    # Setup Scheduler Components
    config = CircuitBreakerConfig(failure_threshold=1) # Fast fail for test
    cb = WorkerCircuitBreaker(redis, config)
    rerouter = RequestRerouter(redis)
    
    # 1. Register the worker (Simulate startup)
    await cb.register_worker(worker_id)
    await redis.set(f"worker:heartbeat:{worker_id}", "1", ex=2)
    
    # 2. Simulate worker fetching a Premium task mid-batch
    in_flight_task = {"request_id": "req-vip-1", "priority": "premium"}
    await redis.lpush(f"queue:processing:{worker_id}", json.dumps(in_flight_task))
    
    # Verify initial healthy state
    assert await cb.get_state(worker_id) == WorkerState.HEALTHY
    assert await redis.llen(f"queue:processing:{worker_id}") == 1
    
    # 3. THE CRASH: Simulate time passing and heartbeat expiring
    await redis.delete(f"worker:heartbeat:{worker_id}")
    
    # 4. The Scheduler runs its cycle
    heartbeat_exists = await redis.exists(f"worker:heartbeat:{worker_id}")
    
    if not heartbeat_exists:
        # Scheduler logs a failure
        new_state = await cb.record_heartbeat_failure(worker_id)
        
        # Scheduler isolates the worker and reroutes
        if new_state == WorkerState.UNHEALTHY:
            await rerouter.recover_orphaned_requests(worker_id)
            
    # 5. Verify the Recovery
    # The worker must be isolated
    assert await cb.get_state(worker_id) == WorkerState.UNHEALTHY
    
    # The processing queue must be cleared
    assert await redis.llen(f"queue:processing:{worker_id}") == 0
    
    # The task must not be dropped; it must be back in the Premium queue
    assert await redis.llen("queue:premium") == 1
    recovered_task = json.loads(await redis.rpop("queue:premium"))
    assert recovered_task["request_id"] == "req-vip-1"
    assert recovered_task["requeued"] is True