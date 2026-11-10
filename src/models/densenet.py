"""
PulmoVision — DenseNet121 Architecture for 2D/2.5D CT Classification
Provides feature reuse via dense connectivity.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class PulmoDenseNet121(nn.Module):
    """
    DenseNet121 adapted for PulmoVision CT slice classification.
    
    Args:
        in_channels: 1 for 2D or 3 for 2.5D (default: 3).
        num_classes: Number of target categories (default: 2).
        pretrained: If True, loads official ImageNet weights.
        dropout: Dropout rate before the final classifier layer.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = densenet121(weights=weights)

        # 1. Adapt input layer if in_channels != 3
        # In DenseNet, features.conv0 is the initial 7x7 Conv2d
        if in_channels != 3:
            old_conv = self.model.features.conv0
            self.model.features.conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    self.model.features.conv0.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # 2. Replace classification head (classifier: in_features -> num_classes)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
