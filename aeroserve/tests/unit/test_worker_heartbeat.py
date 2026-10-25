import asyncio
import pytest
from worker.heartbeat import HeartbeatEmitter

@pytest.mark.asyncio
async def test_heartbeat_emitter_lifecycle(redis):
    worker_id = "worker-test-1"
    emitter = HeartbeatEmitter(redis, worker_id, heartbeat_interval_sec=0.1, ttl_sec=2)
    
    # Start heartbeat coroutine
    await emitter.start()
    await asyncio.sleep(0.15) # Wait for first pulse
    
    # Check registration and pulse
    is_registered = await redis.sismember("registered_workers", worker_id)
    assert is_registered
    
    pulse_exists = await redis.exists(f"worker:heartbeat:{worker_id}")
    assert pulse_exists == 1
    
    # Stop and verify it exits cleanly
    await emitter.stop()
    assert emitter.is_running is False