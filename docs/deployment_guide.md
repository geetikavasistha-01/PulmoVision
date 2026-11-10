# PulmoVision — Cloud Deployment & Optimization Guide

This guide details how to deploy the PulmoVision Streamlit application with minimal latency and high availability.

---

## 1. Measured Inference Latency & Benchmarks

Empirical performance measured on our production checkpoint (`models/best_model.pth`):

| Hardware Mode | Cold-Start (Model Load + 1st Pass) | Warm-Start Latency (Mean) | P50 (Median) | P95 | Throughput |
|---|---|---|---|---|---|
| **Apple Silicon Metal (MPS)** | 10.2 s (Shader compilation) | **140.1 ms** | **111.1 ms** | 302.4 ms | **7.1 scans/sec** |
| **Standard CPU** | **1.16 s** | **899.9 ms** | **836.5 ms** | 1255.5 ms | **1.1 scans/sec** |

> **Key takeaway:** On standard cloud containers without a GPU (e.g. Streamlit Cloud standard tier), warm inference runs in $< 900\text{ ms}$, delivering a responsive real-time user experience without requiring expensive GPU instances.

---

## 2. Deployment Architecture: Streamlit Community Cloud

```mermaid
flowchart LR
    A["GitHub Repository\n(main branch)"] --> B["Streamlit Cloud Engine"]
    B --> C["Install requirements-deploy.txt"]
    C --> D["Launch app/streamlit_app.py"]
    D --> E["Load & Cache Model\n@st.cache_resource"]
    E --> F["Public Web Endpoint\nhttps://pulmovision.streamlit.app"]
```

### Steps to Deploy:
1. Push all changes to GitHub: `git push origin main`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository: `geetikavasistha-01/PulmoVision`.
4. Set:
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** `3.10` or `3.11`
5. Click **Deploy**.

---

## 3. Docker Deployment (Containerized Serving)

Create a lightweight container using the provided `requirements-deploy.txt`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run:
```bash
docker build -t pulmovision:latest .
docker run -p 8501:8501 pulmovision:latest
```

---

## 4. Production Optimizations Implemented

| Optimization | Implementation | Benefit |
|---|---|---|
| **Resource Caching** | `@st.cache_resource` on `load_predictor()` | Weights loaded once; zero reload overhead on user interaction |
| **Headless UI** | `.streamlit/config.toml` (`headless = true`, `gatherUsageStats = false`) | Strips browser telemetries and disables CORS restrictions |
| **Pruned Requirements** | `requirements-deploy.txt` ($< 300\text{MB}$) | Eliminates developer/test dependencies (`pytest`, `six`, etc.) |
| **Single-Slice Clamp Fallback** | `CTSliceStacker` edge replication | Prevents crashes when users upload single 2D slices |

