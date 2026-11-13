# PulmoVision — Container Deployment Guide (Stage 3A)

## Overview

This guide provides reproducible containerization instructions for deploying the PulmoVision CT Sequence Intelligence Workstation on standard Linux and cloud container runtimes without manual Python environment management.

---

## 1. Container Architecture & Security

- **Base Image:** `python:3.11-slim` (Debian Bookworm)
- **Non-Root User:** `pulmo:1000` (Least-privilege operational model)
- **Port:** `8501` (Bound to `0.0.0.0`)
- **Health Check Endpoint:** `http://localhost:8501/_stcore/health`
- **Model Checkpoint:** `models/stage2_bilstm_deployment.pth` (Pre-validated SHA-256: `0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed`)
- **Pretrained Weights Cache:** EfficientNet-B0 ImageNet-1K weights are pre-baked during image build into `TORCH_HOME=/app/.cache/torch` for offline air-gapped readiness.
- **Data Quarantine Security:** Raw patient images (`data/raw/`) and the sealed test cohort metadata (`data/metadata/Annotation/CSV/test1.csv`) are strictly excluded via `.dockerignore`.

---

## 2. Quick Start

### Building the Image

```bash
docker build -t pulmovision:latest .
```

### Running the Container

```bash
docker run -d \
  --name pulmovision \
  -p 8501:8501 \
  --memory="4g" \
  --cpus="2" \
  pulmovision:latest
```

Open your browser to `http://localhost:8501` to access the workstation.

### Running with Docker Compose

```bash
docker compose up -d
```

To stop:
```bash
docker compose down
```

---

## 3. Container Verification & Health Monitoring

### Healthcheck Status
Check if the application container has booted and is healthy:
```bash
docker inspect --format='{{json .State.Health.Status}}' pulmovision
```
Expected output: `"healthy"`.

### Direct Endpoint Query
```bash
curl -f http://localhost:8501/_stcore/health
```
Expected response: HTTP 200 `ok`.

---

## 4. Hardware & Inference Benchmarking

### CPU vs Apple Silicon (M1) Latency Comparison

| Hardware Target | Execution Runtime | Model Load (Cold) | End-to-End Sequence Latency (Warm) | Per-Slice Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Apple M1 Mac** | Local Python (MPS Metal) | 0.22 s | **140.1 ms** | ~46.7 ms / slice |
| **Standard CPU** | Container / Host CPU | 0.18 s | **585.7 ms** | ~195.2 ms / slice |

*Note: Benchmarked using the 3-slice permitted development series `patient100_SR_2`. The container falls back to standard CPU execution safely without requiring Apple Metal or CUDA drivers.*

---

## 5. Deployment Limitations & Boundaries

1. **Research Prototype:** PulmoVision is an educational and research prototype. It is not approved by regulatory bodies (FDA/CE) for primary diagnostic decision-making.
2. **CPU Inference:** Standard container instances run inference on CPU. While responsive (<600 ms for typical CT slice triplets), long sequences (>50 slices) will scale linearly with slice count.
3. **Data Isolation:** The container operates using bundled demonstration sequences (`data/demo_samples/`) and user-uploaded TIFF/PNG CT slices. Uploaded files exist only in runtime memory and are never persisted to disk.
