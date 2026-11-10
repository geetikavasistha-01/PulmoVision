"""
PulmoVision — Unified Model Factory
Provides a single interface to instantiate, configure, and inspect any of our
Stage 1 architectures (Custom CNN, ResNet50, EfficientNet-B0, DenseNet121).
"""

from typing import Dict, Any, Tuple
import torch.nn as nn
from src.models.custom_cnn import PulmoCustomCNN
from src.models.resnet import PulmoResNet50
from src.models.efficientnet import PulmoEfficientNetB0
from src.models.densenet import PulmoDenseNet121


MODEL_REGISTRY = {
    "custom_cnn": PulmoCustomCNN,
    "resnet50": PulmoResNet50,
    "efficientnet_b0": PulmoEfficientNetB0,
    "densenet121": PulmoDenseNet121,
}


def get_model(
    model_name: str,
    in_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Instantiates an adapted architecture by registry name.
    
    Args:
        model_name: One of 'custom_cnn', 'resnet50', 'efficientnet_b0', 'densenet121'.
        in_channels: 1 for 2D, 3 for 2.5D.
        num_classes: Number of classification categories (default: 2).
        pretrained: Whether to load ImageNet pretrained weights (ignored for custom_cnn).
        dropout: Classification head dropout probability.
    
    Returns:
        Adapted nn.Module ready for forward pass and training.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[key]
    
    if key == "custom_cnn":
        return model_cls(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    else:
        return model_cls(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )


def compare_model_parameters(in_channels: int = 3, num_classes: int = 2) -> Dict[str, Dict[str, int]]:
    """
    Utility returning total and trainable parameter counts for all registered models.
    """
    comparison = {}
    for name in MODEL_REGISTRY:
        # Load without pretrained download for instantaneous parameter counting
        model = get_model(name, in_channels=in_channels, num_classes=num_classes, pretrained=False)
        total, trainable = model.count_parameters()
        comparison[name] = {
            "total_params": total,
            "trainable_params": trainable
        }
    return comparison

