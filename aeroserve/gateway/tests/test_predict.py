import json

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app, lifespan


@pytest.fixture
def anyio_backend():
    """Required for pytest.mark.anyio to know which async backend to use."""
    return "asyncio"

@pytest.fixture
async def async_client():
    """Initializes the AsyncClient and natively triggers the Redis lifespan."""
    
    async with lifespan(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client

@pytest.mark.anyio
async def test_health_check_endpoint(async_client):
    """Unit: Health check returns service status and connection states."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert "redis_connected" in response.json()

@pytest.mark.anyio
async def test_predict_classification_and_validation(async_client):
    """Unit: Request validation and default priority tier handling."""
    resp = await async_client.post(
        "/predict",
        json={"model_name": "isic-classifier", "payload": {"image_b64": "test_data"}}
    )
    assert resp.status_code == 202
    assert resp.json()["queue"] == "standard"

    resp_premium = await async_client.post(
        "/predict",
        json={
            "model_name": "isic-classifier", 
            "payload": {"image_b64": "test_data"},
            "priority": "premium"
        }
    )
    assert resp_premium.status_code == 202
    assert resp_premium.json()["queue"] == "premium"

    resp_invalid = await async_client.post(
        "/predict",
        json={
            "model_name": "isic-classifier", 
            "payload": {"image_b64": "test_data"},
            "priority": "ultra-fast"
        }
    )
    assert resp_invalid.status_code == 422

@pytest.mark.anyio
async def test_gateway_to_redis_integration(async_client):
    """Integration: Full route test proving item lands with correct shape in Redis."""
    target_queue = "background"
    
    # Because the lifespan is active, app.state.redis is fully accessible
    redis_client = app.state.redis
    
    await redis_client.delete(target_queue)
    
    test_payload = {"features": [1.0, 2.5, 3.8]}
    response = await async_client.post(
        "/predict",
        json={
            "model_name": "tabular-nn",
            "payload": test_payload,
            "priority": target_queue
        }
    )
    assert response.status_code == 202
    assigned_id = response.json()["request_id"]
    
    raw_item = await redis_client.rpop(target_queue)
    assert raw_item is not None
    
    enqueued_data = json.loads(raw_item)
    assert enqueued_data["request_id"] == assigned_id
    assert enqueued_data["priority"] == "background"
    assert enqueued_data["model_name"] == "tabular-nn"