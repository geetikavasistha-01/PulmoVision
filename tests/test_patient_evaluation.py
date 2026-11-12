"""
Automated unit tests for patient-level aggregated metric calculations.
"""

import pytest
from src.evaluation.metrics import compute_classification_metrics, compute_patient_level_metrics


def test_slice_vs_patient_unweighted_equality():
    """
    Ensures a patient with 100 slices does NOT dominate a patient with 10 slices.
    """
    # Patient A: 100 slices, all True Normal (0). Model incorrectly predicts 2 slices as COVID (1).
    pA_targets = [0] * 100
    pA_preds = [0] * 98 + [1] * 2  # 2% false positives
    pA_ids = ["patient_A"] * 100

    # Patient B: 10 slices, all True COVID (1). Model correctly predicts all 10 slices as COVID (1).
    pB_targets = [1] * 10
    pB_preds = [1] * 10
    pB_ids = ["patient_B"] * 10

    slice_preds = pA_preds + pB_preds
    slice_targets = pA_targets + pB_targets
    patient_ids = pA_ids + pB_ids

    # 1. At the SLICE level: 110 total slices
    slice_res = compute_classification_metrics(slice_targets, slice_preds)
    # Accuracy = (98 + 10) / 110 = 98.2%
    assert pytest.approx(slice_res["accuracy"], 0.01) == 0.982

    # 2. At the PATIENT level: 2 total patients (Patient A and Patient B)
    # With threshold=0.10 (10% positive slices needed to call patient positive):
    # Patient A has 2% positive slices -> Patient A predicted 0 (True Negative)
    # Patient B has 100% positive slices -> Patient B predicted 1 (True Positive)
    patient_res = compute_patient_level_metrics(
        slice_preds=slice_preds,
        slice_targets=slice_targets,
        patient_ids=patient_ids,
        strategy="positive_slice_ratio",
        threshold=0.10
    )

    assert patient_res["total_patients"] == 2
    assert patient_res["accuracy"] == 1.0  # Both patients correctly diagnosed
    assert patient_res["macro_f1"] == 1.0
    assert patient_res["tp"] == 1
    assert patient_res["tn"] == 1
    assert patient_res["fp"] == 0
    assert patient_res["fn"] == 0


def test_patient_aggregation_strategies():
    slice_preds = [1, 1, 0, 0, 0]  # 40% positive
    slice_targets = [1, 1, 1, 1, 1]  # COVID patient
    slice_scores = [0.8, 0.7, 0.2, 0.1, 0.2]  # mean score = 0.40
    patient_ids = ["patient_X"] * 5

    # Strategy 1: positive_slice_ratio with threshold 0.30 -> should be 1
    res1 = compute_patient_level_metrics(slice_preds, slice_targets, patient_ids, strategy="positive_slice_ratio", threshold=0.30)
    assert res1["per_patient_summary"]["patient_x"]["pred_label"] == 1

    # Strategy 2: positive_slice_ratio with threshold 0.50 -> should be 0
    res2 = compute_patient_level_metrics(slice_preds, slice_targets, patient_ids, strategy="positive_slice_ratio", threshold=0.50)
    assert res2["per_patient_summary"]["patient_x"]["pred_label"] == 0

    # Strategy 3: mean_prob with threshold 0.35 -> should be 1 (0.40 >= 0.35)
    res3 = compute_patient_level_metrics(slice_preds, slice_targets, patient_ids, slice_scores=slice_scores, strategy="mean_prob", threshold=0.35)
    assert res3["per_patient_summary"]["patient_x"]["pred_label"] == 1

