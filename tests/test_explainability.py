"""
Automated unit tests for PulmoVision Grad-CAM explainability module.
"""

import pytest
import torch
import numpy as np
from src.models.model_factory import get_model
from src.explainability.gradcam import GradCAM


def test_gradcam_heatmap_generation():
    model = get_model("efficientnet_b0", in_channels=3, num_classes=2, pretrained=False)
    cam = GradCAM(model=model)

    x = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, confidence = cam.generate_heatmap(x)

    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
    assert pred_class in [0, 1]
    assert 0.0 <= confidence <= 1.0


def test_gradcam_overlay():
    dummy_slice = np.random.uniform(0.0, 1.0, (224, 224))
    dummy_heatmap = np.random.uniform(0.0, 1.0, (224, 224))

    overlay = GradCAM.overlay_heatmap(dummy_heatmap, dummy_slice)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
