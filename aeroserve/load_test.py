import requests
import concurrent.futures
import time

def send_request(i):
    try:
        res = requests.post(
            "http://localhost:8000/predict",
            json={
                "model_name": "iris_model",
                "priority": "standard",
                "user_tier": "user",
                "payload": {"features": [5.1, 3.5, 1.4, 0.2]}
            }
        )
        return res.status_code
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    start_time = time.time()
    print("Firing 250 concurrent requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(send_request, range(250)))
    print(f"Finished in {time.time() - start_time:.2f} seconds.")