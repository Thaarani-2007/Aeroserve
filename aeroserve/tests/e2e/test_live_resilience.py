import asyncio
import httpx
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AeroServe.E2E")

# Update these to match your local setup
GATEWAY_URL = "http://localhost:8000/predict"  # Your FastAPI endpoint
WORKER_CONTAINER_NAME = "aeroserve-worker-1"   # Check `docker ps` for the exact name

async def trigger_prediction(client: httpx.AsyncClient, request_id: str):
    """Sends a prediction request and waits for the response."""
    
    # UPDATED PAYLOAD: Matches the FastAPI PredictRequest schema exactly
    payload = {
        "model_name": "test_cnn_model", 
        "priority": "premium",
        "payload": {
            "image_data": "base64_encoded_image_string_here"
        }
    }
    
    start_time = time.time()
    logger.info(f"Sending request {request_id} to Gateway...")
    
    try:
        response = await client.post(GATEWAY_URL, json=payload, timeout=15.0)
        elapsed = time.time() - start_time
        
        assert response.status_code == 200, f"Gateway returned {response.status_code}: {response.text}"
        logger.info(f"SUCCESS: Request {request_id} completed in {elapsed:.2f}s despite worker death!")
        return True
    except httpx.ReadTimeout:
        logger.error(f"FAIL: Request {request_id} timed out. The request was dropped or gateway hung.")
        return False
    except Exception as e:
        logger.error(f"FAIL: Request {request_id} resulted in error: {e}")
        return False

async def kill_worker_container():
    """Simulates a catastrophic container failure mid-batch."""
    # Wait briefly to ensure the request has hit the worker's processing queue
    await asyncio.sleep(0.5) 
    logger.warning(f"Executing Chaos: Killing container {WORKER_CONTAINER_NAME}...")
    
    # Subprocess calls the host's Docker daemon to forcefully kill the container
    process = subprocess.run(
        ["docker", "kill", WORKER_CONTAINER_NAME], 
        capture_output=True, 
        text=True
    )
    
    if process.returncode == 0:
        logger.warning(f"Container {WORKER_CONTAINER_NAME} successfully killed.")
    else:
        logger.error(f"Failed to kill container: {process.stderr}")

async def run_e2e_resilience_test():
    """Runs the E2E test flow."""
    logger.info("Starting Live E2E Resilience Test...")
    
    async with httpx.AsyncClient() as client:
        # Fire the HTTP request and the container kill command concurrently
        results = await asyncio.gather(
            trigger_prediction(client, "req-e2e-chaos-1"),
            kill_worker_container()
        )
        
        request_success = results[0]
        if request_success:
            logger.info("E2E Test Passed: The scheduler successfully rescued and rerouted the task.")
        else:
            logger.error("E2E Test Failed: The system dropped the task.")

if __name__ == "__main__":
    asyncio.run(run_e2e_resilience_test())