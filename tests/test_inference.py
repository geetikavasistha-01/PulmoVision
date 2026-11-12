"""
Automated unit tests for PulmoVision standalone inference engine and predictor.
"""

import pytest
import numpy as np
import torch
from src.inference.predictor import PulmoPredictor


def test_predictor_with_dummy_slices():
    # Test initialization
    predictor = PulmoPredictor(checkpoint_path="models/best_model.pth", device="cpu")

    # Mock 16-bit CT slice
    slice_z = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)
    slice_prev = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)
    slice_next = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)

    # Full 3-slice 2.5D prediction
    res = predictor.predict(slice_z=slice_z, slice_prev=slice_prev, slice_next=slice_next)

    assert res["prediction"] in ["Normal", "COVID-19"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "Normal" in res["probabilities"]
    assert "COVID-19" in res["probabilities"]
    assert res["display_slice"].shape == (224, 224)
    assert res["heatmap"].shape == (224, 224)
    assert res["overlay"].shape == (224, 224, 3)
    assert res["latency_ms"] > 0.0


def test_predictor_single_slice_fallback():
    predictor = PulmoPredictor(checkpoint_path="models/best_model.pth", device="cpu")
    single_slice = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)

    # Call with only single slice (prev and next default to None -> clamp fallback)
    res = predictor.predict(slice_z=single_slice)
    assert res["prediction"] in ["Normal", "COVID-19"]
    assert res["overlay"].shape == (224, 224, 3)

