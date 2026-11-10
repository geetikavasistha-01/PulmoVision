"""
PulmoVision — Comprehensive Model Evaluation Metrics
Computes clinical and statistical classification metrics:
Accuracy, Precision, Recall (Sensitivity), Specificity, Macro F1, and Balanced Accuracy.
"""

from typing import Dict, List, Any
import numpy as np


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes diagnostic performance metrics for binary classification (0 = Normal, 1 = COVID).
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))

    total = len(y_t)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Sensitivity / Recall (Class 1)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (Class 0 True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Positive Predictive Value / Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 Score (Class 1)
    f1_covid = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # F1 Score (Class 0)
    prec_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_normal = (2 * prec_normal * specificity) / (prec_normal + specificity) if (prec_normal + specificity) > 0 else 0.0

    macro_f1 = (f1_covid + f1_normal) / 2.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_covid": f1_covid,
        "f1_normal": f1_normal,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

