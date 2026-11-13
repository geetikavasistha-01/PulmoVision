"""
PulmoVision — Streamlit Community Cloud Deployment Readiness Verification.
Validates entrypoint shim, dependency constraints, checkpoint integrity,
and CPU sequence inference without raw dataset dependencies.
"""
import os
import sys
import hashlib
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_CKPT_PATH = REPO_ROOT / "models" / "stage2_bilstm_deployment.pth"
EXPECTED_CKPT_SHA256 = "0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed"


def test_root_entrypoint_shim_exists_and_points_to_canonical():
    """Verify root entrypoint shims (streamlit_app.py and app.py) exist and point to app/streamlit_app.py."""
    canonical = REPO_ROOT / "app" / "streamlit_app.py"
    assert canonical.is_file(), "Canonical app/streamlit_app.py missing!"
    
    for shim_name in ["streamlit_app.py", "app.py"]:
        shim = REPO_ROOT / shim_name
        assert shim.is_file(), f"Root {shim_name} shim missing!"
        content = shim.read_text()
        assert "app/streamlit_app.py" in content or "run_path" in content


def test_requirements_hygiene():
    """Verify requirements.txt is clean of development and conflicting packages."""
    req_file = REPO_ROOT / "requirements.txt"
    assert req_file.is_file(), "requirements.txt missing!"
    lines = [line.strip().lower() for line in req_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    
    # Must not contain dev-only packages
    for forbidden in ["pytest", "kaggle", "kagglesdk", "torchaudio"]:
        assert not any(forbidden in line for line in lines), f"Forbidden package {forbidden} found in requirements.txt!"
        
    # Must use headless opencv only
    has_headless = any("opencv-python-headless" in line for line in lines)
    has_standard_opencv = any("opencv-python==" in line or "opencv-python>=" in line for line in lines)
    assert has_headless, "opencv-python-headless must be in requirements.txt"
    assert not has_standard_opencv, "Conflicting opencv-python must NOT be in requirements.txt"


def test_deployment_checkpoint_present_and_checksum_valid():
    """Verify stage2_bilstm_deployment.pth is present with bit-for-bit verified SHA-256."""
    assert DEPLOYMENT_CKPT_PATH.is_file(), f"Missing deployment checkpoint: {DEPLOYMENT_CKPT_PATH}"
    
    hasher = hashlib.sha256()
    with open(DEPLOYMENT_CKPT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_CKPT_SHA256, (
        f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_CKPT_SHA256}\nActual:   {actual_hash}"
    )


def test_cpu_sequence_predictor_inference():
    """Verify SequencePredictor initializes cleanly on CPU and runs inference on demo samples."""
    from src.inference.sequence_predictor import SequencePredictor
    
    predictor = SequencePredictor(checkpoint_path=str(DEPLOYMENT_CKPT_PATH), device="cpu")
    assert predictor.device.type == "cpu"
    assert predictor.checkpoint_loaded is True
    assert predictor.checksum_verified is True
    
    demo_files = sorted((REPO_ROOT / "data" / "demo_samples").glob("*.tif"))
    assert len(demo_files) > 0, "No demo sample TIFFs found for validation!"
    
    res = predictor.predict_sequence(demo_files)
    assert "predicted_class" in res
    assert res["predicted_class"] in ["Normal", "COVID-19"]
    assert 0.0 <= res["covid_19_probability"] <= 1.0
    assert 0.0 <= res["normal_probability"] <= 1.0
    assert res["inference_latency_ms"] > 0
