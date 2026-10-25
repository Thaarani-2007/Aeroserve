import os
import sys
import time
import math
import logging
import subprocess
import requests
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Autoscaler]: %(message)s")
logger = logging.getLogger("AeroServe.Autoscaler")

PROMETHEUS_URL = "http://prometheus:9090"
POLL_INTERVAL_SEC = 5

def get_dynamic_scaling_limits():
    """Scans all model_config.json files for zero-touch scaling limits."""
    base_min_workers = 1
    aggregate_max_workers = 1
    avg_target_queue = 15
    
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        return base_min_workers, aggregate_max_workers, avg_target_queue
        
    targets = []
    for folder in os.listdir(models_dir):
        if folder.startswith(".tmp_"): continue
        
        config_path = os.path.join(models_dir, folder, "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    autoscaling = config.get("autoscaling", {})
                    
                    m_max = autoscaling.get("max_workers", 1)
                    if m_max > aggregate_max_workers:
                        aggregate_max_workers = m_max
                        
                    targets.append(autoscaling.get("target_queue_per_worker", 15))
            except Exception as e:
                logger.error(f"Failed to read scaling config from {folder}: {e}")
                
    if targets:
        avg_target_queue = sum(targets) // len(targets)
        
    return base_min_workers, aggregate_max_workers, avg_target_queue

class QueueAutoscaler:
    def __init__(self):
        self.min_w, self.max_w, self.target_q = get_dynamic_scaling_limits()
        self.current_workers = self.min_w
        self.last_scale_time = 0.0

    def query_prometheus(self, query: str) -> float:
        try:
            response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
            response.raise_for_status()
            results = response.json().get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
            return 0.0
        except Exception:
            return 0.0

    def execute_scale(self, target_workers: int, action: str):
        logger.info(f"Action: {action} | Scaling workers from {self.current_workers} -> {target_workers}")
        try:
            subprocess.run(
                ["docker", "compose", "up", "-d", "--scale", f"worker={target_workers}"],
                capture_output=True, text=True, check=True, cwd=PROJECT_ROOT 
            )
            self.current_workers = target_workers
            self.last_scale_time = time.time()
            logger.info(f"Scale successful. New worker count: {target_workers}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Scale command failed! STDERR: {e.stderr}")

    def evaluate(self):
        now = time.time()
        time_since_last_scale = now - self.last_scale_time

        self.min_w, self.max_w, self.target_q = get_dynamic_scaling_limits()

        total_queue_depth = self.query_prometheus("sum(aeroserve_queue_depth)")
        oldest_age = self.query_prometheus("max(aeroserve_oldest_request_age_seconds)")

        depth_desired = math.ceil(total_queue_depth / self.target_q) if self.target_q > 0 else self.min_w
        
        age_desired = 0
        if oldest_age > 2.0: 
            age_desired = self.current_workers + 1
            
        raw_desired = max(depth_desired, age_desired)
        desired_workers = max(self.min_w, min(raw_desired, self.max_w))

        # FIX: Using the uppercase configuration settings
        if desired_workers > self.current_workers:
            if time_since_last_scale >= settings.SCALE_UP_COOLDOWN_SEC:
                logger.info("UPWARD PRESSURE DETECTED.")
                self.execute_scale(desired_workers, "SCALE UP")
            else:
                cooldown_left = settings.SCALE_UP_COOLDOWN_SEC - time_since_last_scale
                logger.info(f"Scale up in cooldown ({cooldown_left:.0f}s left)")

        elif desired_workers < self.current_workers:
            current_capacity = self.current_workers * self.target_q
            hysteresis_threshold = current_capacity * settings.SCALE_DOWN_HYSTERESIS_FACTOR

            if total_queue_depth < hysteresis_threshold and oldest_age == 0:
                if time_since_last_scale >= settings.SCALE_DOWN_COOLDOWN_SEC:
                    logger.info("DOWNWARD PRESSURE DETECTED.")
                    self.execute_scale(desired_workers, "SCALE DOWN")
                else:
                    cooldown_left = settings.SCALE_DOWN_COOLDOWN_SEC - time_since_last_scale
                    logger.info(f"Scale down in cooldown ({cooldown_left:.0f}s left)")

    def run(self):
        logger.info("AeroServe Queue-Driven Autoscaler Started.")
        while True:
            self.evaluate()
            time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    autoscaler = QueueAutoscaler()
    try:
        autoscaler.run()
    except KeyboardInterrupt:
        logger.info("Autoscaler shut down by user.")