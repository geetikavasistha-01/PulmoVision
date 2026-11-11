"""
PulmoVision — Stage 2 Nested Cross-Validation Infrastructure
Implements 5-fold outer cross-validation with inner stratified validation splits
for strict, unbiased out-of-fold (OOF) model selection and threshold tuning.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_classification_metrics


def ensure_cv_5fold_manifest(
    dev_csv_path: str = "data/metadata/Annotation/CSV/train1.csv",
    manifest_path: str = "data/metadata/splits/cv_5fold_manifest.json",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ensures that the reproducible 5-fold outer cross-validation split manifest exists on disk.
    If missing, deterministically generates it from train1.csv using StratifiedKFold(seed=42).
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
            
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df = pd.read_csv(dev_csv_path)
    df["patient_id"] = df["filename"].apply(lambda f: re.search(r"patient\d+", f.lower()).group(0))
    patient_df = df.groupby("patient_id")["class"].first().reset_index()
    patient_df["label_bin"] = patient_df["class"].apply(
        lambda l: 1 if "covid" in str(l).lower() or str(l) == "1" else 0
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_df["patient_id"], patient_df["label_bin"])):
        train_patients = patient_df.iloc[train_idx]["patient_id"].tolist()
        val_patients = patient_df.iloc[val_idx]["patient_id"].tolist()
        val_labels = patient_df.iloc[val_idx]["label_bin"].tolist()
        splits[f"fold_{fold+1}"] = {
            "train_patients": sorted(train_patients),
            "val_patients": sorted(val_patients),
            "val_covid_count": sum(val_labels),
            "val_normal_count": len(val_labels) - sum(val_labels),
        }

    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def generate_nested_splits(
    outer_manifest_path: str = "data/metadata/splits/cv_5fold_manifest.json",
    dev_csv_path: str = "data/metadata/Annotation/CSV/train1.csv",
    seed_offset: int = 100
) -> Dict[str, Any]:
    """
    Constructs inner stratified splits from outer training patients for each outer fold.
    
    Returns dict structure:
    {
      "fold_1": {
         "outer_train_patients": [...],
         "outer_val_patients": [...],
         "inner_train_patients": [...],
         "inner_val_patients": [...]
      }, ...
    }
    """
    outer_splits = ensure_cv_5fold_manifest(dev_csv_path=dev_csv_path, manifest_path=outer_manifest_path)
        
    df_dev = pd.read_csv(dev_csv_path)
    df_dev["patient_id"] = df_dev["filename"].apply(lambda f: re.search(r"patient\d+", f.lower()).group(0))
    patient_labels = df_dev.groupby("patient_id")["class"].first().to_dict()
    patient_bin = {pid: (1 if "covid" in str(lbl).lower() or str(lbl) == "1" else 0) for pid, lbl in patient_labels.items()}

    nested_manifest = {}
    all_outer_val_patients = []

    for fold_idx, (fold_key, fold_info) in enumerate(outer_splits.items(), start=1):
        outer_train = fold_info["train_patients"]
        outer_val = fold_info["val_patients"]
        all_outer_val_patients.extend(outer_val)

        # Labels for outer train patients
        outer_train_labels = [patient_bin[pid] for pid in outer_train]

        # Inner stratified 80/20 split (5-fold split, take first fold as inner val)
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_offset + fold_idx)
        inner_train_idx, inner_val_idx = next(inner_skf.split(outer_train, outer_train_labels))

        inner_train_pids = [outer_train[i] for i in inner_train_idx]
        inner_val_pids = [outer_train[i] for i in inner_val_idx]

        # Invariant checks
        assert len(set(inner_train_pids).intersection(set(inner_val_pids))) == 0, f"Inner overlap in {fold_key}"
        assert len(set(outer_train).intersection(set(outer_val))) == 0, f"Outer overlap in {fold_key}"
        assert set(inner_train_pids).union(set(inner_val_pids)) == set(outer_train)

        nested_manifest[fold_key] = {
            "outer_train_patients": outer_train,
            "outer_val_patients": outer_val,
            "inner_train_patients": inner_train_pids,
            "inner_val_patients": inner_val_pids,
            "outer_val_covid_count": sum(patient_bin[pid] for pid in outer_val),
            "outer_val_normal_count": len(outer_val) - sum(patient_bin[pid] for pid in outer_val),
            "inner_val_covid_count": sum(patient_bin[pid] for pid in inner_val_pids),
            "inner_val_normal_count": len(inner_val_pids) - sum(patient_bin[pid] for pid in inner_val_pids),
        }

    assert len(all_outer_val_patients) == 122, f"Total outer val patients {len(all_outer_val_patients)} != 122"
    assert len(set(all_outer_val_patients)) == 122, "Duplicate patients across outer validation sets"

    return nested_manifest


def compute_bootstrap_confidence_intervals(
    y_true: List[int],
    y_pred: List[int],
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% empirical bootstrap confidence intervals (2.5th and 97.5th percentiles)
    for Macro F1, Sensitivity, and Specificity over pooled OOF predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    boot_f1 = []
    boot_sens = []
    boot_spec = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]

        m = compute_classification_metrics(yt_sample.tolist(), yp_sample.tolist())
        boot_f1.append(m["macro_f1"])
        boot_sens.append(m["sensitivity_recall"])
        boot_spec.append(m["specificity"])

    return {
        "macro_f1_ci_95": (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))),
        "sensitivity_ci_95": (float(np.percentile(boot_sens, 2.5)), float(np.percentile(boot_sens, 97.5))),
        "specificity_ci_95": (float(np.percentile(boot_spec, 2.5)), float(np.percentile(boot_spec, 97.5))),
    }


def find_optimal_threshold(
    y_true: List[int],
    y_probs: List[float],
    threshold_grid: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Searches inner validation probabilities to find classification threshold tau*
    that maximizes inner Macro F1.
    """
    if threshold_grid is None:
        threshold_grid = [float(t) for t in np.linspace(0.10, 0.90, 17)]
        
    best_tau = 0.50
    best_f1 = -1.0
    
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    
    for tau in threshold_grid:
        preds = (y_probs_arr >= tau).astype(int).tolist()
        m = compute_classification_metrics(y_true, preds)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_tau = tau
            
    return best_tau, best_f1
