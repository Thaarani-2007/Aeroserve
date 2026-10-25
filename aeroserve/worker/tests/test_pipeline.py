import asyncio
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestClient")

API_URL = "http://localhost:8000/predict"

async def send_request(client: httpx.AsyncClient, req_id: int, priority: str):
    """Sends a single request and measures the exact round-trip latency."""
    payload = {
        "model_name": "test_model",
        "priority": priority,
        "payload": {"test_id": req_id}
    }
    start_time = time.time()
    
    try:
        response = await client.post(API_URL, json=payload, timeout=12.0)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            logger.info(f"✅ [Req {req_id:03d} | {priority[:4].upper()}] Success in {latency:.0f}ms")
            return {"status": "success", "latency": latency, "priority": priority}
        else:
            logger.error(f"❌ [Req {req_id:03d} | {priority[:4].upper()}] Failed: HTTP {response.status_code}")
            return {"status": "error", "code": response.status_code}
            
    except httpx.ReadTimeout:
        logger.error(f"❌ [Req {req_id:03d} | {priority[:4].upper()}] Gateway Timeout (Client Side)")
        return {"status": "timeout"}

async def run_scenario_1_basic(client):
    logger.info("--- SCENARIO 1: Basic End-to-End ---")
    await send_request(client, 1, "premium")
    await asyncio.sleep(1)

async def run_scenario_2_mixed_batch(client):
    logger.info("\n--- SCENARIO 2: Mixed Batch (High-Water Mark Timeout) ---")
    # Send 1 Premium, 2 Standard, 2 Background simultaneously.
    # The worker should group them into ONE batch of 5.
    # The whole batch must execute in <50ms because of the Premium request!
    tasks = [
        send_request(client, 10, "premium"),
        send_request(client, 11, "standard"),
        send_request(client, 12, "standard"),
        send_request(client, 13, "background"),
        send_request(client, 14, "background")
    ]
    await asyncio.gather(*tasks)
    await asyncio.sleep(1)

async def run_scenario_3_max_batch_overflow(client):
    logger.info("\n--- SCENARIO 3: MAX_BATCH_SIZE (Overflow) ---")
    # MAX_BATCH_SIZE is 16. We will send 20 requests at the exact same time.
    # We should see the worker process one batch of 16, then a batch of 4.
    tasks = [send_request(client, 20 + i, "standard") for i in range(20)]
    results = await asyncio.gather(*tasks)
    
    successes = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Result: {successes}/20 requests processed successfully.")
    await asyncio.sleep(1)

async def run_scenario_4_wrr_fairness(client):
    logger.info("\n--- SCENARIO 4: WRR Fairness Under Heavy Load ---")
    # Flood the system with 30 Premium requests, but sneak in 5 Background requests.
    # Even though Premium is flooded, the 5:3:1 WRR should ensure Background requests
    # get processed WITHOUT having to wait for all 30 Premium requests to finish.
    tasks = []
    for i in range(30):
        tasks.append(send_request(client, 100 + i, "premium"))
    for i in range(5):
        tasks.append(send_request(client, 200 + i, "background"))
        
    await asyncio.gather(*tasks)

async def main():
    async with httpx.AsyncClient() as client:
        await run_scenario_1_basic(client)
        await run_scenario_2_mixed_batch(client)
        await run_scenario_3_max_batch_overflow(client)
        await run_scenario_4_wrr_fairness(client)

if __name__ == "__main__":
    asyncio.run(main())