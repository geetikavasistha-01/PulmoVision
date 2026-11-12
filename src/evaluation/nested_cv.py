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
    that maximizes inner Macro F1 with a deterministic tie-breaking rule favoring tau = 0.50.
    
    Tie-breaking rule:
    1. Maximum Macro F1.
    2. If tied, threshold closest to canonical prior tau = 0.50.
    3. If equidistant (e.g. 0.45 vs 0.55), pick the higher threshold for conservative specificity.
    """
    if threshold_grid is None:
        threshold_grid = [float(t) for t in np.linspace(0.20, 0.80, 13)]
        
    best_tau = 0.50
    best_f1 = -1.0
    
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    
    for tau in threshold_grid:
        preds = (y_probs_arr >= tau).astype(int).tolist()
        m = compute_classification_metrics(y_true, preds)
        f1 = m["macro_f1"]
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
        elif abs(f1 - best_f1) < 1e-6:
            # Tie-break towards 0.50
            if abs(tau - 0.50) < abs(best_tau - 0.50):
                best_tau = tau
            elif abs(abs(tau - 0.50) - abs(best_tau - 0.50)) < 1e-6:
                best_tau = max(tau, best_tau)
            
    return best_tau, best_f1


def compute_patient_balanced_loss(
    logits: torch.Tensor,
    series_patient_indices: torch.Tensor,
    patient_labels: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes patient-balanced loss:
    1. Evaluates individual series losses using unreduced class-weighted cross-entropy.
    2. Aggregates series losses belonging to the same patient via exact mean.
    3. Computes the arithmetic mean across patient-level losses.
    
    Guarantees:
    - Every patient in the minibatch contributes exactly 1/B to the objective.
    - A patient with N_p series contributes exactly 1/B * (1/N_p * sum(series_losses)).
    - A patient with 1 series contributes exactly 1/B * series_loss.
    """
    B = patient_labels.size(0)
    series_labels = patient_labels[series_patient_indices]
    unreduced_loss = criterion(logits, series_labels) # [M]

    patient_losses = torch.zeros(B, device=logits.device, dtype=logits.dtype)
    for p_idx in range(B):
        p_mask = (series_patient_indices == p_idx)
        patient_losses[p_idx] = unreduced_loss[p_mask].mean()

    return patient_losses.mean(), patient_losses


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Trains sequence model for one epoch over patient-balanced batches.
    Enforces EQUAL PATIENT-LEVEL OPTIMIZATION:
    All series for a patient are evaluated together; patient loss is the mean over
    their series, and batch loss is the unweighted mean across the B patients.
    """
    model.train()
    total_loss = 0.0
    num_patients = 0

    for batch in loader:
        optimizer.zero_grad()
        x = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        # Handle models with or without lengths argument
        try:
            logits = model(x, mask=mask, lengths=lengths)
        except TypeError:
            logits = model(x, mask=mask)

        if "series_patient_indices" in batch and "patient_labels" in batch:
            series_patient_indices = batch["series_patient_indices"].to(device)
            patient_labels = batch["patient_labels"].to(device)
            loss, p_losses = compute_patient_balanced_loss(
                logits, series_patient_indices, patient_labels, criterion
            )
            b_pts = patient_labels.size(0)
            loss.backward()
            optimizer.step()
            total_loss += p_losses.sum().item()
            num_patients += b_pts
        else:
            labels = batch["labels"].to(device)
            patient_weights = batch.get("patient_weights", torch.ones_like(labels, dtype=torch.float32)).to(device)
            unreduced_loss = criterion(logits, labels)
            weighted_loss = unreduced_loss * patient_weights
            loss = weighted_loss.sum() / patient_weights.sum().clamp(min=1e-6)
            loss.backward()
            optimizer.step()
            total_loss += weighted_loss.sum().item()
            num_patients += len(labels)

    return total_loss / max(num_patients, 1)


@torch.no_grad()
def evaluate_patient_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates model on patient-balanced sequence batches, returning patient-aggregated probabilities.
    For each patient, computes the mean predicted probability across all their series.
    """
    model.eval()
    patient_results: Dict[str, Dict[str, Any]] = {}

    for batch in loader:
        x = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        try:
            logits = model(x, mask=mask, lengths=lengths)
        except TypeError:
            logits = model(x, mask=mask)

        probs = torch.softmax(logits, dim=1)[:, 1] # COVID-19 probability [M]

        if "series_patient_indices" in batch and "patient_labels" in batch:
            series_patient_indices = batch["series_patient_indices"].to(device)
            patient_labels = batch["patient_labels"]
            patient_ids = batch["patient_ids"]

            B = patient_labels.size(0)
            for p_idx in range(B):
                p_mask = (series_patient_indices == p_idx)
                p_prob = probs[p_mask].mean().item()
                pid = patient_ids[p_idx]
                patient_results[pid] = {
                    "patient_id": pid,
                    "true_label": int(patient_labels[p_idx].item()),
                    "pred_prob": float(p_prob),
                    "num_series": int(p_mask.sum().item()),
                }
        else:
            labels = batch["labels"].tolist()
            patient_ids = batch["patient_ids"]
            for pid, prob, lbl in zip(patient_ids, probs.cpu().tolist(), labels):
                patient_results.setdefault(pid, {"probs": [], "label": lbl})["probs"].append(prob)

    if not patient_results or "probs" not in next(iter(patient_results.values())):
        return patient_results

    # Fallback aggregation for non-patient-balanced loader
    formatted = {}
    for pid, data in patient_results.items():
        formatted[pid] = {
            "patient_id": pid,
            "true_label": data["label"],
            "pred_prob": float(np.mean(data["probs"])),
            "num_series": len(data["probs"]),
        }
    return formatted


def run_nested_cv_benchmark(
    model_name: str,
    cache_path: str = "data/processed/dev_patient_features_effnet_b0.pt",
    outer_manifest_path: str = "data/metadata/splits/cv_5fold_manifest.json",
    dev_csv_path: str = "data/metadata/Annotation/CSV/train1.csv",
    max_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    patience: int = 5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes complete nested 5-fold cross-validation for a specified model architecture.
    
    Guarantees:
    - Inner validation only for checkpoint selection and threshold search.
    - Outer validation patients evaluated exactly once with frozen weights and frozen threshold.
    - Full suite of 9 metrics reported per fold and pooled across all 122 OOF predictions.
    - 95% bootstrap confidence intervals over pooled OOF predictions.
    """
    from src.models.cnn_bilstm import get_stage2_model
    from src.data.sequence_dataset import CachedEmbeddingDataset, collate_sequence_embeddings
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score
    import copy

    if "test1" in dev_csv_path.lower() or "test1" in outer_manifest_path.lower():
        raise ValueError("Safety violation: Nested CV must never access test1.csv!")

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    nested_splits = generate_nested_splits(
        outer_manifest_path=outer_manifest_path,
        dev_csv_path=dev_csv_path,
        seed_offset=100
    )

    fold_results = {}
    all_oof_predictions: Dict[str, Dict[str, Any]] = {}

    for fold_idx, (fold_key, fold_info) in enumerate(nested_splits.items(), start=1):
        # Set deterministic seeds per fold
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        inner_train_pids = fold_info["inner_train_patients"]
        inner_val_pids = fold_info["inner_val_patients"]
        outer_val_pids = fold_info["outer_val_patients"]

        # Datasets indexed directly by patient to enforce equal patient-level optimization
        from src.data.sequence_dataset import PatientSequenceDataset, collate_patient_balanced_batch

        inner_train_ds = PatientSequenceDataset(cache, patient_ids=inner_train_pids)
        inner_val_ds = PatientSequenceDataset(cache, patient_ids=inner_val_pids)
        outer_val_ds = PatientSequenceDataset(cache, patient_ids=outer_val_pids)

        inner_train_loader = DataLoader(
            inner_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_balanced_batch
        )
        inner_val_loader = DataLoader(
            inner_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_balanced_batch
        )
        outer_val_loader = DataLoader(
            outer_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_balanced_batch
        )

        # Class weights calculated exclusively from inner-training PATIENT counts
        inner_train_labels = [item["label"] for item in inner_train_ds]
        n_covid = sum(inner_train_labels)
        n_norm = len(inner_train_labels) - n_covid
        w_norm = len(inner_train_labels) / (2.0 * max(n_norm, 1))
        w_covid = len(inner_train_labels) / (2.0 * max(n_covid, 1))
        class_weights = torch.tensor([w_norm, w_covid], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        model = get_stage2_model(model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_inner_f1 = -1.0
        best_state_dict = None
        epochs_no_improve = 0

        # Inner training loop
        for epoch in range(1, max_epochs + 1):
            train_loss = train_epoch(model, inner_train_loader, optimizer, criterion, device)
            inner_val_preds = evaluate_patient_predictions(model, inner_val_loader, device)

            y_inner_true = [v["true_label"] for v in inner_val_preds.values()]
            y_inner_probs = [v["pred_prob"] for v in inner_val_preds.values()]
            y_inner_binary = [int(p >= 0.50) for p in y_inner_probs]
            m_epoch = compute_classification_metrics(y_inner_true, y_inner_binary)

            if m_epoch["macro_f1"] > best_inner_f1:
                best_inner_f1 = m_epoch["macro_f1"]
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Restore best inner checkpoint
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Threshold selection on inner-validation predictions only
        inner_val_preds = evaluate_patient_predictions(model, inner_val_loader, device)
        y_inner_true = [v["true_label"] for v in inner_val_preds.values()]
        y_inner_probs = [v["pred_prob"] for v in inner_val_preds.values()]
        optimal_tau, inner_best_f1 = find_optimal_threshold(
            y_inner_true, y_inner_probs, threshold_grid=[float(t) for t in np.linspace(0.20, 0.80, 13)]
        )

        # Freeze model and tau*, evaluate outer validation ONCE
        outer_val_preds = evaluate_patient_predictions(model, outer_val_loader, device)
        y_outer_true = [v["true_label"] for v in outer_val_preds.values()]
        y_outer_probs = [v["pred_prob"] for v in outer_val_preds.values()]
        y_outer_pred = [int(p >= optimal_tau) for p in y_outer_probs]

        fold_metrics = compute_classification_metrics(y_outer_true, y_outer_pred)
        try:
            fold_metrics["roc_auc"] = float(roc_auc_score(y_outer_true, y_outer_probs))
        except ValueError:
            fold_metrics["roc_auc"] = 0.5
        try:
            fold_metrics["pr_auc"] = float(average_precision_score(y_outer_true, y_outer_probs))
        except ValueError:
            fold_metrics["pr_auc"] = 0.0

        fold_metrics["optimal_threshold"] = optimal_tau
        fold_metrics["inner_best_macro_f1"] = inner_best_f1

        fold_results[fold_key] = fold_metrics

        for pid, data in outer_val_preds.items():
            prob = data["pred_prob"]
            all_oof_predictions[pid] = {
                "patient_id": pid,
                "true_label": data["true_label"],
                "pred_prob": prob,
                "pred_binary": int(prob >= optimal_tau),
                "threshold_applied": optimal_tau,
                "outer_fold": fold_key,
            }

    # Verify all 122 patients received OOF predictions
    assert len(all_oof_predictions) == 122, f"OOF count {len(all_oof_predictions)} != 122"

    pooled_true = [all_oof_predictions[pid]["true_label"] for pid in sorted(all_oof_predictions.keys())]
    pooled_prob = [all_oof_predictions[pid]["pred_prob"] for pid in sorted(all_oof_predictions.keys())]
    pooled_pred = [all_oof_predictions[pid]["pred_binary"] for pid in sorted(all_oof_predictions.keys())]

    pooled_metrics = compute_classification_metrics(pooled_true, pooled_pred)
    try:
        pooled_metrics["roc_auc"] = float(roc_auc_score(pooled_true, pooled_prob))
    except ValueError:
        pooled_metrics["roc_auc"] = 0.5
    try:
        pooled_metrics["pr_auc"] = float(average_precision_score(pooled_true, pooled_prob))
    except ValueError:
        pooled_metrics["pr_auc"] = 0.0

    # 95% Bootstrap CIs over pooled predictions
    bootstrap_cis = compute_bootstrap_confidence_intervals(pooled_true, pooled_pred, n_bootstraps=1000, seed=seed)

    # Compute cross-fold mean and std
    cross_fold_summary = {}
    for metric_key in ["macro_f1", "roc_auc", "pr_auc", "sensitivity_recall", "specificity", "balanced_accuracy", "precision", "accuracy"]:
        vals = [fold_results[fk][metric_key] for fk in fold_results]
        cross_fold_summary[metric_key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
        }

    return {
        "model_name": model_name,
        "per_fold_metrics": fold_results,
        "cross_fold_summary": cross_fold_summary,
        "pooled_oof_metrics": pooled_metrics,
        "bootstrap_cis_95": bootstrap_cis,
        "oof_predictions": all_oof_predictions,
    }
