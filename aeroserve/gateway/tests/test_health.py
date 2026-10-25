import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app, lifespan


@pytest.fixture
def anyio_backend():
    """Required for pytest.mark.anyio to know which async backend to use."""
    return "asyncio"

@pytest.mark.anyio
async def test_health_check():
    # Explicitly trigger the lifespan to initialize app.state.redis
    
        async with lifespan(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert "status" in response.json()