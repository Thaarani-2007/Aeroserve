import concurrent.futures
import requests
import time
import random

# Ensure this matches the port FastAPI will run on
URL = "http://localhost:8000/predict"
PRIORITIES = ["premium", "standard", "background"]
WEIGHTS = [20, 50, 30] # 20% premium, 50% standard, 30% background

def send_request(req_id: int):
    priority = random.choices(PRIORITIES, weights=WEIGHTS)[0]
    payload = {
        "model_name": "sentiment_analyzer",
        "priority": priority,
        "user_tier": "user",
        "payload": {"text": f"Simulating faculty demo request {req_id} for AeroServe."}
    }
    
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            print(f"[{priority.upper():<10}] HTTP 200 | Latency: {latency:.2f}s | Result: {response.json().get('inference_result')}")
        else:
            print(f"[{priority.upper():<10}] HTTP {response.status_code} | Latency: {latency:.2f}s | Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"[{priority.upper():<10}] FAILED: Connection Refused (Is the Gateway running?)")

def run_demo(total_requests=100, concurrency=15):
    print(f"🚀 Launching {total_requests} requests at {URL} with concurrency {concurrency}...")
    
    # Use ThreadPoolExecutor to simulate multiple users hitting the endpoint simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        concurrent.futures.wait(futures)
        
    print("✅ Demo load test complete.")

if __name__ == "__main__":
    # Simulate 100 requests with 15 concurrent connections
    run_demo(total_requests=100, concurrency=15)