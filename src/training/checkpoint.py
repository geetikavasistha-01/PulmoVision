"""
PulmoVision — Model Checkpointing System
Saves and restores model weights, optimizer state, and training metadata.
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
    model_name: str = "efficientnet_b0",
) -> None:
    """
    Saves state dictionary, hyperparameters, and validation metrics to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
    }
    torch.save(state, filepath)
    print(f"[Checkpoint] Saved best model to: {filepath} (Epoch {epoch}, Val F1: {metrics.get('val_f1', 0.0):.4f})")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Loads saved weights into model and optimizer.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded '{checkpoint.get('model_name')}' from {filepath} (Epoch {checkpoint.get('epoch')})")
    return checkpoint
