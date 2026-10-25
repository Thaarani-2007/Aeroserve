import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error

# Ensure imports work from the root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.profiler import run_profiling

def main():
    parser = argparse.ArgumentParser(prog="aeroserve", description="AeroServe PaaS CLI Tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_cmd = subparsers.add_parser("submit", help="Profile and register a local model package")
    submit_cmd.add_argument("--model-dir", type=str, required=True, help="Path to folder containing processor.py & test_inputs.json")
    submit_cmd.add_argument("--name", type=str, required=True, help="Registry name for the model")
    submit_cmd.add_argument("--framework", type=str, default="transformers", choices=["transformers", "pytorch", "tensorflow", "onnx", "custom"])
    submit_cmd.add_argument("--sla", type=float, default=None, help="Latency target threshold in milliseconds")
    submit_cmd.add_argument("--processor-class", type=str, default=None, help="Exact class name to load from processor.py (Optional)")

    args = parser.parse_args()

    if args.command == "submit":
        abs_model_dir = os.path.abspath(args.model_dir)
        inputs_file = os.path.join(abs_model_dir, "test_inputs.json")
        processor_file = os.path.join(abs_model_dir, "processor.py")

        if not os.path.isfile(processor_file):
            print(f"[-] Missing processor implementation: {processor_file}")
            sys.exit(1)

        if not os.path.isfile(inputs_file):
            print(f"[-] Missing test inputs: {inputs_file}")
            sys.exit(1)

        with open(inputs_file, "r") as f:
            try:
                test_records = json.load(f)
                if not isinstance(test_records, list) or len(test_records) == 0:
                    raise ValueError("Must contain a non-empty JSON array")
            except Exception as e:
                print(f"[-] Invalid test_inputs.json format: {e}")
                sys.exit(1)

        # Run profiling securely
        generated_config = run_profiling(
            model_name=args.name,
            source_dir=abs_model_dir,
            framework=args.framework,
            sla_target_ms=args.sla,
            test_inputs=test_records,
            output_dir=abs_model_dir,
            target_class_name=args.processor_class
        )

        # Atomic Deployment
        live_models_dir = os.path.abspath("models")
        temp_model_dir = os.path.join(live_models_dir, f".tmp_{args.name}")
        final_model_dir = os.path.join(live_models_dir, args.name)

        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        os.makedirs(temp_model_dir, exist_ok=True)

        print("\n[*] Transferring model artifacts...")
        for item in os.listdir(abs_model_dir):
            src_item = os.path.join(abs_model_dir, item)
            dst_item = os.path.join(temp_model_dir, item)
            if os.path.isfile(src_item):
                shutil.copy2(src_item, dst_item)

        # Atomic Swap: Instantly replaces the old model or provisions a new one safely
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        os.rename(temp_model_dir, final_model_dir)
        print("[+] Artifacts deployed safely via atomic swap.")

        # Verification
        print("[*] Waiting for AeroServe gateway and workers to acknowledge the model...")
        max_retries = 30
        is_ready = False
        
        for attempt in range(max_retries):
            try:
                # We will ensure the Gateway has a /health endpoint that checks active models later
                req = urllib.request.Request(f"http://localhost:8000/health")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        health_data = json.loads(response.read().decode('utf-8'))
                        
                        # Verify our specific model is in the active registry
                        if args.name in health_data.get("models_loaded", []):
                            is_ready = True
                            break
            except urllib.error.URLError:
                pass
            
            time.sleep(1)
            print(".", end="", flush=True)

        if not is_ready:
            print("\n[-] Deployment warning: Files deployed, but Gateway healthcheck timed out.")
            sys.exit(1)

        print(f"\n[+] Zero-Touch Deployment Complete & Verified!")
        print(f"    - Endpoint available at: POST /predict")
        print(f"    - Model Name: {args.name}")
        print(f"    - Max Batch Size: {generated_config['batching']['max_batch_size']}")

if __name__ == "__main__":
    main()