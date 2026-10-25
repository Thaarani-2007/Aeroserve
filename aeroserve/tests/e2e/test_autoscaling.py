import asyncio
import httpx
import time
import logging
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AeroServe.LoadTest")

GATEWAY_URL = "http://localhost:8000/predict"
TOTAL_REQUESTS = 100

async def fire_request(client: httpx.AsyncClient, req_id: int):
    """Fires a single inference request."""
    payload = {
        # 1. UPDATED to hit the zero-touch registry
        "model_name": "resnet50_classifier", 
        "priority": "premium",
        "payload": {
            "image_data": f"mock_base64_data_{req_id}"
        }
    }
    try:
        # 2. UPDATED timeout to 60.0 to survive autoscaler cold starts
        response = await client.post(GATEWAY_URL, json=payload, timeout=60.0)
        if response.status_code == 200:
            logger.debug(f"Request {req_id} succeeded.")
        else:
            logger.warning(f"Request {req_id} failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Request {req_id} error: {e}")

async def run_load_test():
    logger.info(f"Initiating Load Test: Firing {TOTAL_REQUESTS} requests...")
    start_time = time.time()
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as client:
        tasks = []
        for i in range(TOTAL_REQUESTS):
            tasks.append(fire_request(client, i))
            # ADD THIS: A tiny stagger to prevent local TCP socket exhaustion
            
        await asyncio.gather(*tasks)
        
    elapsed = time.time() - start_time
    logger.info(f"Load Test Complete! Sent {TOTAL_REQUESTS} requests in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(run_load_test())