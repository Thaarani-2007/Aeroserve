import pytest
import pytest_asyncio
import fakeredis.aioredis

@pytest_asyncio.fixture
async def redis():
    """Provides a fresh in-memory Redis instance for each test."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    yield client
    await client.flushall()
    await client.aclose()