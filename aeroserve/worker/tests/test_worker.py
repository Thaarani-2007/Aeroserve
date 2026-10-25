import pytest

from models.aeroserve_preprocessor import AeroServeCustomProcessor
from worker.main import process_task


class ExplodingProcessor(AeroServeCustomProcessor):
    def execute(self, payload: dict) -> dict:
        _ = payload["missing_key"] 
        return {"result": "success"}

@pytest.fixture
def mock_hybrid_processor():
    return ExplodingProcessor()

@pytest.fixture
def mock_session():
    class DummySession:
        def run(self, *args, **kwargs):
            return [[0.99]]
    return DummySession()

@pytest.mark.asyncio
async def test_worker_survives_bad_input(mock_hybrid_processor, mock_session):
    bad_task = {"request_id": "req-123", "payload": {"corrupted": "data"}}
    
    result = await process_task(mock_hybrid_processor, mock_session, bad_task)
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["request_id"] == "req-123"