"""
PulmoVision — ResNet50 Architecture for 2D/2.5D CT Classification
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class PulmoResNet50(nn.Module):
    """
    ResNet50 adapted for PulmoVision CT slice classification.
    
    Args:
        in_channels: 1 for 2D or 3 for 2.5D (default: 3).
        num_classes: Number of target categories (default: 2).
        pretrained: If True, loads official ImageNet weights.
        dropout: Dropout rate before the final classification head.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = resnet50(weights=weights)

        # 1. Adapt input layer if in_channels != 3
        if in_channels != 3:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # If 1-channel, average the 3 RGB channel weights
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    self.model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # 2. Replace final classification head (fc: in_features -> num_classes)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
