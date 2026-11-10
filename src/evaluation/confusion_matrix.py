"""
PulmoVision — Confusion Matrix Visualizer and Evaluator
"""

import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.metrics import compute_classification_metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ["Normal", "COVID-19"],
    output_path: str = "reports/figures/confusion_matrix.png",
    title: str = "PulmoVision 2.5D Evaluation — Confusion Matrix",
) -> np.ndarray:
    """
    Renders and saves a clinical confusion matrix with counts and percentages.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = compute_classification_metrics(y_true, y_pred)
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="bold")

    # Annotate numbers inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / np.sum(cm[i])) * 100 if np.sum(cm[i]) > 0 else 0
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    ax.set_ylabel("True Ground Truth Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {output_path}")
    return cm


if __name__ == "__main__":
    # Test sample
    y_t = [0] * 30 + [1] * 10
    y_p = [0] * 28 + [1] * 2 + [0] * 2 + [1] * 8
    plot_confusion_matrix(y_t, y_p)
