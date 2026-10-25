import threading
import requests
import time

def fire_request():
    url = "http://localhost:8000/predict"
    payload = {
        "model_name": "resnet50_classifier",
        "priority": "background",
        "payload": {"data": "burst_test"}
    }
    while True: # Keep firing continuously
        try:
            requests.post(url, json=payload)
        except Exception:
            pass
        time.sleep(0.1) # Small delay to prevent crashing your own machine

print("Starting sustained load test (Press Ctrl+C to stop)...")
threads = []
for _ in range(20):
    t = threading.Thread(target=fire_request, daemon=True)
    t.start()
    threads.append(t)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped!")