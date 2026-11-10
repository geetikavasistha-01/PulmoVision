"""
Automated unit tests for PulmoVision CT preprocessing pipeline.
"""

import os
import pytest
import numpy as np
import torch
from src.data.preprocessing import CTPreprocessor, compare_normalizations

SAMPLE_TIFF = "data/interim/sample_audit/TrainValidation/137covid_patient100_SR_2_IM00028.tif"

def test_ct_preprocessor_validation_mode():
    preprocessor = CTPreprocessor(target_size=(224, 224), is_training=False)
    
    if os.path.exists(SAMPLE_TIFF):
        tensor = preprocessor(SAMPLE_TIFF)
    else:
        # Fallback dummy 16-bit CT slice
        dummy_ct = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)
        tensor = preprocessor(dummy_ct)
        
    assert isinstance(tensor, torch.Tensor), "Output must be a torch.Tensor"
    assert tensor.shape == (1, 224, 224), f"Expected shape (1, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert tensor.min() >= 0.0, f"Values must be >= 0.0, got min {tensor.min()}"
    assert tensor.max() <= 1.0, f"Values must be <= 1.0, got max {tensor.max()}"
    assert not torch.isnan(tensor).any(), "Tensor must not contain NaNs"
    assert not torch.isinf(tensor).any(), "Tensor must not contain Infs"

def test_ct_preprocessor_training_mode():
    preprocessor = CTPreprocessor(target_size=(224, 224), is_training=True)
    dummy_ct = np.random.randint(0, 2200, (512, 512), dtype=np.uint16)
    tensor = preprocessor(dummy_ct)
    
    assert tensor.shape == (1, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_compare_normalizations():
    dummy_ct = np.random.uniform(0, 2200, (100, 100)).astype(np.float32)
    res = compare_normalizations(dummy_ct)
    
    assert "linear_min_max" in res
    assert "percentile_1_99" in res
    assert "lung_window" in res
    for name, arr in res.items():
        assert arr.shape == (100, 100)
        assert 0.0 <= arr.min() <= arr.max() <= 1.0
