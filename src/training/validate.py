"""
PulmoVision — Validation Loop
Evaluates model on validation batches, computing loss, accuracy, and Macro F1.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Runs a deterministic validation pass without gradients.
    
    Returns:
        Dict with 'val_loss', 'val_acc', 'val_f1', 'predictions', 'targets'.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1).cpu().tolist()
            targets = batch_y.cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)

    n_samples = len(all_targets)
    val_loss = total_loss / max(1, n_samples)
    val_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / max(1, n_samples)

    # Compute Macro F1 (Balanced per-class F1)
    val_f1 = compute_macro_f1(all_preds, all_targets, num_classes=2)

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "predictions": all_preds,
        "targets": all_targets,
    }


def compute_macro_f1(preds: List[int], targets: List[int], num_classes: int = 2) -> float:
    """
    Computes unweighted mean F1 score across all classes.
    """
    f1_scores = []
    for c in range(num_classes):
        tp = sum(p == c and t == c for p, t in zip(preds, targets))
        fp = sum(p == c and t != c for p, t in zip(preds, targets))
        fn = sum(p != c and t == c for p, t in zip(preds, targets))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / max(1, len(f1_scores))

