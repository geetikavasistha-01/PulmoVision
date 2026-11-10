"""
Automated unit tests for PulmoVision neural network architectures.
"""

import pytest
import torch
from src.models.custom_cnn import PulmoCustomCNN


def test_custom_cnn_2d_forward():
    batch_size = 4
    model = PulmoCustomCNN(in_channels=1, num_classes=2)
    x = torch.randn(batch_size, 1, 224, 224)
    out = model(x)

    assert out.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaNs"
    assert not torch.isinf(out).any(), "Model output contains Infs"


def test_custom_cnn_25d_forward():
    batch_size = 2
    model = PulmoCustomCNN(in_channels=3, num_classes=2)
    x = torch.randn(batch_size, 3, 224, 224)
    out = model(x)

    assert out.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaNs"


def test_parameter_count():
    model = PulmoCustomCNN(in_channels=1, num_classes=2)
    total, trainable = model.count_parameters()
    
    assert total > 0
    assert total == trainable
    # Baseline custom CNN should be lightweight (< 200k params)
    assert total < 200_000, f"Expected lightweight baseline, got {total} params"
    print(f"\nCustom CNN Parameters: {total:,} total, {trainable:,} trainable")

