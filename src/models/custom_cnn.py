"""
PulmoVision — Baseline Custom CNN Architecture
A modular, lightweight convolutional network supporting both 2D (1-channel)
and 2.5D (multi-channel) CT slice classification.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PulmoCustomCNN(nn.Module):
    """
    Custom baseline CNN for PulmoVision.
    
    Args:
        in_channels: Number of input channels (1 for single slice 2D, 3 for 2.5D triplet).
        num_classes: Number of target prediction categories (default: 2 -> Normal vs COVID).
        dropout: Dropout probability before the classification head.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor: 224x224 -> 112x112 -> 56x56 -> 28x28
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # Global spatial aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape:  [B, in_channels, 224, 224]
        Output shape: [B, num_classes] (raw logits)
        """
        feats = self.features(x)
        pooled = self.global_pool(feats)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """
        Returns (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

