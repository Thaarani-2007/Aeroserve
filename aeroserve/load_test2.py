import asyncio
import aiohttp
import time

URL = "http://localhost:8000/predict"
PAYLOAD = {
    "model_name": "iris_model",
    "priority": "standard",
    "user_tier": "user",
    "payload": {"features": [5.1, 3.5, 1.4, 0.2]}
}
DURATION_SECONDS = 40

async def send_requests(session):
    requests_sent = 0
    end_time = time.time() + DURATION_SECONDS
    while time.time() < end_time:
        try:
            async with session.post(URL, json=PAYLOAD) as response:
                await response.read()
                requests_sent += 1
        except Exception:
            pass
    return requests_sent

async def main():
    print(f"🚀 Firing asynchronous blast for {DURATION_SECONDS} seconds...")
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=500)) as session:
        tasks = [send_requests(session) for _ in range(200)]
        results = await asyncio.gather(*tasks)
    print(f"✅ Fired {sum(results)} requests!")

if __name__ == "__main__":
    asyncio.run(main())