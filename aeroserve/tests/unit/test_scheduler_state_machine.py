import pytest
from scheduler.state_machine import WorkerCircuitBreaker, CircuitBreakerConfig, WorkerState

@pytest.mark.asyncio
async def test_circuit_breaker_transitions(redis):
    config = CircuitBreakerConfig(failure_threshold=2, consecutive_success_threshold=2)
    cb = WorkerCircuitBreaker(redis, config)
    worker_id = "worker-cb-1"
    
    await cb.register_worker(worker_id)
    assert await cb.get_state(worker_id) == WorkerState.HEALTHY
    
    # 1st failure - should remain healthy
    await cb.record_heartbeat_failure(worker_id)
    assert await cb.get_state(worker_id) == WorkerState.HEALTHY
    
    # 2nd failure - threshold reached, should trip to unhealthy
    await cb.record_heartbeat_failure(worker_id)
    assert await cb.get_state(worker_id) == WorkerState.UNHEALTHY
    
    # 1st success - should transition to draining/half-open
    await cb.record_heartbeat_success(worker_id)
    assert await cb.get_state(worker_id) == WorkerState.DRAINING
    
    # 2nd success - should fully restore to healthy
    await cb.record_heartbeat_success(worker_id)
    assert await cb.get_state(worker_id) == WorkerState.HEALTHY