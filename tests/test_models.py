"""
Automated unit tests for PulmoVision neural network architectures and model factory.
"""

import pytest
import torch
from src.models.custom_cnn import PulmoCustomCNN
from src.models.model_factory import get_model, compare_model_parameters, MODEL_REGISTRY


@pytest.mark.parametrize("model_name", ["custom_cnn", "resnet50", "efficientnet_b0", "densenet121"])
def test_all_models_25d_forward(model_name):
    batch_size = 2
    # Pretrained=False for ultra-fast unit testing without network calls
    model = get_model(model_name, in_channels=3, num_classes=2, pretrained=False)
    x = torch.randn(batch_size, 3, 224, 224)
    out = model(x)

    assert out.shape == (batch_size, 2), f"{model_name} forward pass failed: shape is {out.shape}"
    assert not torch.isnan(out).any(), f"{model_name} output contains NaNs"
    assert not torch.isinf(out).any(), f"{model_name} output contains Infs"


@pytest.mark.parametrize("model_name", ["custom_cnn", "resnet50", "efficientnet_b0", "densenet121"])
def test_all_models_2d_forward(model_name):
    batch_size = 2
    model = get_model(model_name, in_channels=1, num_classes=2, pretrained=False)
    x = torch.randn(batch_size, 1, 224, 224)
    out = model(x)

    assert out.shape == (batch_size, 2)
    assert not torch.isnan(out).any()


def test_invalid_model_name_raises_error():
    with pytest.raises(ValueError):
        get_model("transformer_v1")


def test_model_parameter_comparison():
    comp = compare_model_parameters(in_channels=3, num_classes=2)
    assert len(comp) == 4
    for name, stats in comp.items():
        assert stats["total_params"] > 0
        assert stats["total_params"] == stats["trainable_params"]
        print(f"\n{name.upper()}: {stats['total_params']:,} parameters")

    # Verify Custom CNN is smallest and ResNet50 is largest
    assert comp["custom_cnn"]["total_params"] < comp["efficientnet_b0"]["total_params"]
    assert comp["efficientnet_b0"]["total_params"] < comp["resnet50"]["total_params"]
