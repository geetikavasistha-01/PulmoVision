"""
PulmoVision — Latency and Resource Benchmarking Tool
Measures cold-start vs warm-start latency, throughput, and CPU/MPS memory usage.
"""

import time
import numpy as np
import torch
from src.inference.predictor import PulmoPredictor


def benchmark_inference(device_str: str = "cpu", num_runs: int = 50):
    print(f"\n[Benchmark] Running Latency Test on: {device_str.upper()} ({num_runs} runs)")

    # 1. Cold Start: Instantiation + First Forward Pass
    start_cold = time.time()
    predictor = PulmoPredictor(checkpoint_path="models/best_model.pth", device=device_str)
    dummy_slice = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)
    _ = predictor.predict(dummy_slice)
    cold_start_time_ms = (time.time() - start_cold) * 1000.0
    print(f"  • Cold-Start Latency (Model Load + 1st Pass): {cold_start_time_ms:.2f} ms")

    # 2. Warm Start: 50 Iterations
    latencies = []
    for _ in range(num_runs):
        t0 = time.time()
        _ = predictor.predict(dummy_slice)
        latencies.append((time.time() - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"  • Mean Warm Latency: {avg_latency:.2f} ms")
    print(f"  • Median (P50):      {p50:.2f} ms")
    print(f"  • 95th Percentile:   {p95:.2f} ms")
    print(f"  • 99th Percentile:   {p99:.2f} ms")
    print(f"  • Throughput:        {1000.0 / avg_latency:.1f} scans/second")


if __name__ == "__main__":
    # Test CPU
    benchmark_inference("cpu", num_runs=20)
    # Test MPS if available
    if torch.backends.mps.is_available():
        benchmark_inference("mps", num_runs=20)
