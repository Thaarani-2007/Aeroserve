import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import psutil
from typing import Tuple, Type, List, Dict, Any

# Ensure the worker can find the base classes in the root directory
sys.path.insert(0, os.path.abspath("."))
from models.aeroserve_preprocessor import AeroServeProcessor


def load_processor_class(model_dir: str, file_name: str = "processor.py", target_class_name: str = None) -> Tuple[Type[AeroServeProcessor], str, str]:
    """Dynamically imports the developer's processor class."""
    target_path = os.path.join(model_dir, file_name)
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Missing processor implementation at {target_path}")
    
    spec = importlib.util.spec_from_file_location("dynamic_module", target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module specification from {target_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    matched_classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, AeroServeProcessor) and not inspect.isabstract(obj):
            if target_class_name and name == target_class_name:
                return obj, file_name, name
            matched_classes.append((name, obj))
            
    if not matched_classes:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "execute") and obj.__module__ == "dynamic_module" and not inspect.isabstract(obj):
                if target_class_name and name == target_class_name:
                    return obj, file_name, name
                matched_classes.append((name, obj))
                
    if target_class_name:
        raise TypeError(f"Class '{target_class_name}' not found in {file_name}")
        
    if not matched_classes:
        raise TypeError(f"No valid concrete subclass of AeroServeProcessor found in {file_name}")
        
    name, selected_class = matched_classes[0]
    return selected_class, file_name, name

def get_process_memory_mb() -> float:
    """Reads physical RAM or GPU VRAM usage."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))

def run_profiling(
    model_name: str,
    source_dir: str,
    framework: str,
    sla_target_ms: float | None,
    test_inputs: List[Dict[str, Any]],
    output_dir: str,
    target_class_name: str = None
) -> Dict[str, Any]:
    print(f"[*] Profiling model: {model_name}")
    print(f"[*] Reading artifacts from: {source_dir}")
    
    # Pass the target_class_name down to the loader
    ProcessorCls, proc_file, proc_class_name = load_processor_class(source_dir, target_class_name=target_class_name)
    
    print(f"[*] Instantiating processor class '{proc_class_name}'...")
    instance = ProcessorCls()
    

    # Step A: Validate functional output and establish baseline minimum latency
    initial_payload = test_inputs[0]
    
    warmup_start = time.perf_counter()
    test_run_output = instance.execute(initial_payload)
    baseline_latency_ms = (time.perf_counter() - warmup_start) * 1000.0
    
    print(f"[+] Functional verification succeeded. Sample output: {test_run_output}")
    print(f"[+] Absolute minimum hardware latency (Batch 1): {baseline_latency_ms:.2f} ms")

    # Step A.1: Auto-calculate SLA if the developer omitted it
    if sla_target_ms is None or sla_target_ms <= 0:
        sla_target_ms = max(50.0, baseline_latency_ms * 3.0) 
        print(f"[*] No SLA provided. Auto-configured SLA Target to: {sla_target_ms:.0f} ms")

    # Step B: Read available system constraints
    system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    memory_ceiling_mb = system_memory_mb * 0.80

    # Step C: Batch ladder sweep
    batch_ladder = [1, 2, 4, 8, 16, 32, 64]
    benchmark_results: Dict[int, Dict[str, float]] = {}
    chosen_batch_size = 1

    print("\n--- Empirical Benchmark Matrix ---")
    print(f"{'Batch Size':<12}{'Latency (ms)':<16}{'Memory (MB)':<14}{'Status'}")

    for b_size in batch_ladder:
        payloads = [test_inputs[i % len(test_inputs)] for i in range(b_size)]
        
        start_clock = time.perf_counter()
        for p in payloads:
            instance.execute(p)
        latency_duration_ms = (time.perf_counter() - start_clock) * 1000.0

        current_mem_mb = get_process_memory_mb()
        benchmark_results[b_size] = {"latency": latency_duration_ms, "memory": current_mem_mb}

        breached_sla = latency_duration_ms > sla_target_ms
        breached_mem = current_mem_mb > (memory_ceiling_mb / 2.0)

        status_str = "OK"
        if breached_sla:
            status_str = f"EXCEEDED_SLA (>{sla_target_ms:.0f}ms)"
        elif breached_mem:
            status_str = "EXCEEDED_MEMORY_BUDGET"

        print(f"{b_size:<12}{latency_duration_ms:<16.2f}{current_mem_mb:<14.1f}{status_str}")

        if breached_sla or breached_mem:
            break
        chosen_batch_size = b_size

    # Step D: Derive SLAs and Auto-scaling limits based on optimal batch
    optimal_stats = benchmark_results[chosen_batch_size]
    measured_latency_ms = max(1, int(optimal_stats["latency"]))
    measured_memory_mb = max(150.0, optimal_stats["memory"])

    max_concurrent_workers = max(1, int(memory_ceiling_mb / measured_memory_mb))

    base_seconds = measured_latency_ms / 1000.0
    sla_premium_s = max(3.0, round(base_seconds * 10.0, 1))
    sla_standard_s = max(10.0, round(base_seconds * 30.0, 1))
    sla_background_s = max(30.0, round(base_seconds * 100.0, 1))
    
    # Calculate the timeout for the dual-trigger batcher in the worker
    dynamic_batch_timeout_ms = max(15, int(measured_latency_ms / 2))

    config_data = {
        "model_name": model_name,
        "version": "1.0.0",
        "framework": framework,
        "processor": {
            "file": proc_file,
            "class_name": proc_class_name
        },
        "batching": {
            "max_batch_size": chosen_batch_size,
            "batch_timeout_ms": dynamic_batch_timeout_ms
        },
        "slas": {
            "expected_inference_ms": measured_latency_ms,
            "max_queue_wait_s": sla_standard_s,
            "priority_timeouts_s": {
                "premium": sla_premium_s,
                "standard": sla_standard_s,
                "background": sla_background_s
            }
        },
        "autoscaling": {
            "min_workers": 1,
            "max_workers": max_concurrent_workers,
            "target_queue_per_worker": max(4, chosen_batch_size * 2)
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "model_config.json")
    with open(out_file, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"\n[+] Production model configuration written to: {out_file}")
    return config_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroServe Empirical ML Profiler")
    parser.add_argument("--model", type=str, required=True, help="Target model identifier")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing processor.py and test_inputs.json")
    parser.add_argument("--framework", type=str, default="transformers", choices=["transformers", "pytorch", "tensorflow", "onnx", "custom"])
    # Note: SLA is now completely optional. If omitted, the profiler auto-detects it.
    parser.add_argument("--sla", type=float, default=None, help="Max acceptable latency in ms (Optional: Auto-calculated if omitted)")
    
    args = parser.parse_args()
    
    # Standardize input path
    inputs_path = os.path.join(args.model_dir, "test_inputs.json")
    if not os.path.isfile(inputs_path):
        print(f"[-] Missing test inputs: {inputs_path}")
        sys.exit(1)
        
    with open(inputs_path, "r") as f:
        try:
            test_records = json.load(f)
            if not isinstance(test_records, list) or len(test_records) == 0:
                raise ValueError("Must contain a non-empty JSON array")
        except Exception as e:
            print(f"[-] Invalid test_inputs.json format: {e}")
            sys.exit(1)

    run_profiling(
        model_name=args.model,
        source_dir=args.model_dir,
        framework=args.framework,
        sla_target_ms=args.sla,
        test_inputs=test_records,
        output_dir=args.model_dir
    )