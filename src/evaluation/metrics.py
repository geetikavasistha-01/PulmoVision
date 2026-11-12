"""
PulmoVision — Comprehensive Model Evaluation Metrics
Computes both:
1. Slice/triplet-level diagnostic performance.
2. Patient-level aggregated diagnostic performance with unweighted per-patient accounting.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for binomial proportion k/n.
    Particularly robust for small sample sizes (e.g. small positive test cohorts).
    """
    if n == 0:
        return 0.0, 0.0
    import scipy.stats as stats
    z = 1.95996  # 95% standard normal quantile
    p = k / n
    denominator = 1 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denominator
    spread = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denominator
    lower = max(0.0, float(centre - spread))
    upper = min(1.0, float(centre + spread))
    return lower, upper


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Computes diagnostic performance metrics for binary classification (0 = Normal, 1 = COVID).
    Includes Wilson score confidence intervals for Sensitivity and Specificity.
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
    sens_ci_low, sens_ci_high = compute_wilson_ci(tp, tp + fn)

    # Specificity (Class 0 True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    spec_ci_low, spec_ci_high = compute_wilson_ci(tn, tn + fp)

    # Positive Predictive Value / Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 Score (Class 1)
    f1_covid = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # F1 Score (Class 0)
    prec_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_normal = (2 * prec_normal * specificity) / (prec_normal + specificity) if (prec_normal + specificity) > 0 else 0.0

    macro_f1 = (f1_covid + f1_normal) / 2.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    res = {
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "sensitivity_ci_95": (sens_ci_low, sens_ci_high),
        "specificity": specificity,
        "specificity_ci_95": (spec_ci_low, spec_ci_high),
        "precision": precision,
        "f1_covid": f1_covid,
        "f1_normal": f1_normal,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_samples": total,
    }

    # Optional ROC-AUC calculation if continuous scores provided
    if y_scores is not None and len(np.unique(y_t)) > 1:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            res["roc_auc"] = float(roc_auc_score(y_t, y_scores))
            res["pr_auc"] = float(average_precision_score(y_t, y_scores))
        except ImportError:
            pass

    return res


def compute_patient_level_metrics(
    slice_preds: List[int],
    slice_targets: List[int],
    patient_ids: List[str],
    slice_scores: Optional[List[float]] = None,
    strategy: str = "positive_slice_ratio",
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Aggregates slice-level predictions into unweighted patient-level diagnoses.
    Every patient receives weight 1.0 regardless of their total slice count.
    
    Strategies:
      - 'positive_slice_ratio': Patient is Positive if (positive_slices / total_slices) >= threshold.
      - 'mean_prob': Patient is Positive if mean(predicted_probability) >= threshold.
      - 'majority_vote': Patient is Positive if > 50% of slices predict positive.
      
    Args:
        slice_preds: List of slice binary predictions [0, 1]
        slice_targets: List of slice ground truth labels [0, 1]
        patient_ids: List of patient string identifiers matching each slice
        slice_scores: Optional slice-level positive class probabilities
        strategy: Aggregation rule ('positive_slice_ratio', 'mean_prob', 'majority_vote')
        threshold: Decision threshold for positive classification
        
    Returns:
        Dict with patient-level metrics, aggregated counts, and per-patient predictions.
    """
    if not (len(slice_preds) == len(slice_targets) == len(patient_ids)):
        raise ValueError("slice_preds, slice_targets, and patient_ids must have identical lengths")

    # Group slices by patient
    patient_data: Dict[str, Dict[str, Any]] = {}
    for i, pid in enumerate(patient_ids):
        canonical_pid = pid.lower().strip()
        if canonical_pid not in patient_data:
            patient_data[canonical_pid] = {
                "preds": [],
                "targets": [],
                "scores": [],
            }
        patient_data[canonical_pid]["preds"].append(slice_preds[i])
        patient_data[canonical_pid]["targets"].append(slice_targets[i])
        if slice_scores is not None:
            patient_data[canonical_pid]["scores"].append(slice_scores[i])

    patient_true = []
    patient_pred = []
    patient_agg_scores = []
    per_patient_summary = {}

    for pid, data in patient_data.items():
        # Ground truth: target must be identical across all slices of a patient
        true_label = int(np.round(np.mean(data["targets"])))
        n_slices = len(data["preds"])
        pos_slices = sum(p == 1 for p in data["preds"])
        ratio = pos_slices / max(1, n_slices)

        if strategy == "positive_slice_ratio":
            pred_label = 1 if ratio >= threshold else 0
            agg_score = ratio
        elif strategy == "mean_prob":
            if data["scores"]:
                mean_p = float(np.mean(data["scores"]))
                pred_label = 1 if mean_p >= threshold else 0
                agg_score = mean_p
            else:
                pred_label = 1 if ratio >= threshold else 0
                agg_score = ratio
        elif strategy == "majority_vote":
            pred_label = 1 if ratio > 0.50 else 0
            agg_score = ratio
        else:
            raise ValueError(f"Unknown aggregation strategy: '{strategy}'")

        patient_true.append(true_label)
        patient_pred.append(pred_label)
        patient_agg_scores.append(agg_score)

        per_patient_summary[pid] = {
            "true_label": true_label,
            "pred_label": pred_label,
            "slice_count": n_slices,
            "positive_slices": pos_slices,
            "agg_score": agg_score,
        }

    # Compute standard classification metrics at the patient level
    patient_metrics = compute_classification_metrics(patient_true, patient_pred, y_scores=patient_agg_scores)
    patient_metrics["strategy"] = strategy
    patient_metrics["threshold"] = threshold
    patient_metrics["total_patients"] = len(patient_data)
    patient_metrics["per_patient_summary"] = per_patient_summary

    return patient_metrics
