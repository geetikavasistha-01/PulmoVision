# =============================================================================
# PulmoVision — Production Container Image (Stage 3A)
# =============================================================================
# Multi-stage secure build for PulmoVision CT Sequence Intelligence Workstation.
# Runs as non-root user (pulmo:1000) on standard CPU architecture.
# =============================================================================

FROM python:3.11-slim AS runtime

# 1. Environment Configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.cache/torch \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PULMO_DEVICE=cpu

WORKDIR /app

# 2. Install System Dependencies (OpenCV GUI-less bindings & Healthcheck curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Create Non-Root System User & Group
RUN groupadd -g 1000 pulmo && \
    useradd -u 1000 -g pulmo -m -s /bin/bash pulmo

# 4. Install Pinned Deployment Python Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 5. Pre-bake ImageNet-1K EfficientNet-B0 Weights (Air-gap offline readiness)
RUN mkdir -p /app/.cache/torch && \
    python3 -c "import torchvision.models as models; models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)"

# 6. Copy Application Source, Metadata & Checkpoints
COPY . /app

# 7. Secure File Permissions for Non-Root User
RUN chown -R pulmo:pulmo /app

# 8. Drop Privileges to Non-Root User
USER pulmo

# 9. Network Port
EXPOSE 8501

# 10. Container Health Monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 11. Application Entrypoint
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
