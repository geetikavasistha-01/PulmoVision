"""scripts/benchmark_cpu_inference.py - Benchmark PulmoVision Inference on CPU.

Measures cold-start model load time, warm-start inference latency,
per-slice processing time, and memory footprint on standard CPU execution.
"""

import os
import sys
import time
import resource
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sequence_predictor import SequencePredictor
from app.utils.paths import DEMO_SAMPLES_DIR, STAGE2_DEPLOYMENT_CKPT


def run_cpu_benchmark(num_runs: int = 5) -> dict:
    """Executes controlled CPU inference benchmarking on the deployment model."""
    print("=" * 65)
    print(" PulmoVision — CPU Inference Latency & Resource Benchmark")
    print("=" * 65)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Target Device     : CPU (torch.device('cpu'))")
    print(f"Checkpoint Path   : {STAGE2_DEPLOYMENT_CKPT}")

    # 1. Measure Cold-Start Model Load
    t0 = time.perf_counter()
    predictor = SequencePredictor(checkpoint_path=str(STAGE2_DEPLOYMENT_CKPT), device="cpu")
    cold_load_sec = time.perf_counter() - t0
    print(f"Cold-Start Load   : {cold_load_sec:.3f} s")

    # 2. Discover Input Slices
    sample_files = sorted(list(DEMO_SAMPLES_DIR.glob("*.tif")))
    if not sample_files:
        raise FileNotFoundError(f"No demo TIFF files found in {DEMO_SAMPLES_DIR}")
    print(f"Input Sequence    : {len(sample_files)} slices ({[f.name for f in sample_files]})")

    # 3. Warm-up Run
    _ = predictor.predict_sequence(sample_files)

    # 4. Timed Inference Iterations
    latencies = []
    last_res = None
    for run_idx in range(num_runs):
        t_start = time.perf_counter()
        last_res = predictor.predict_sequence(sample_files)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed_ms)
        print(f"  Run {run_idx + 1}/{num_runs}: {elapsed_ms:.1f} ms")

    mean_ms = float(np.mean(latencies))
    p50_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    per_slice_ms = mean_ms / len(sample_files)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024

    print("-" * 65)
    print(f"Warm-Start Mean   : {mean_ms:.1f} ms")
    print(f"Warm-Start P50    : {p50_ms:.1f} ms")
    print(f"Warm-Start P95    : {p95_ms:.1f} ms")
    print(f"Per-Slice Latency : {per_slice_ms:.1f} ms / slice")
    print(f"Peak RSS Memory   : {mem_mb:.1f} MB")
    print(f"Predicted Class   : {last_res['predicted_class']} (Prob: {last_res['covid_19_probability']:.4f})")
    print("=" * 65)

    return {
        "cold_start_load_sec": cold_load_sec,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "per_slice_latency_ms": per_slice_ms,
        "peak_rss_mb": mem_mb,
        "predicted_class": last_res["predicted_class"],
        "covid_19_probability": last_res["covid_19_probability"],
    }


if __name__ == "__main__":
    run_cpu_benchmark()
