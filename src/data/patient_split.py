"""
PulmoVision — Patient-Level Splitting & Partitioning System
Splits development cohort strictly by Patient ID into Train and Validation sets.
Enforces zero-leakage invariants and prevents test set leakage into training loops.
"""

import os
import re
from typing import Dict, List, Tuple, Set, Any
import pandas as pd
import numpy as np


def extract_patient_id(filename: str) -> str:
    """
    Extracts canonical lowercase patient ID from a filename.
    e.g. '137covid_patient100_SR_2_IM00028.tif' -> 'patient100'
    """
    match = re.search(r'(patient\d+)', str(filename), re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract valid patient ID from filename: {filename}")
    return match.group(1).lower()


def create_patient_level_split(
    train_csv_path: str = "data/metadata/Annotation/CSV/train1.csv",
    patient_details_path: str = "data/metadata/Annotation/CSV/Patient_details.csv",
    test_csv_path: str = "data/metadata/Annotation/CSV/test1.csv",
    output_dir: str = "data/metadata/Annotation/CSV",
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Splits train1.csv development partition strictly by Patient ID (stratified by class)
    into train_split.csv (~80%) and val_split.csv (~20%).
    
    Verifies that test1.csv is completely untouched and disjoint from train and val.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Missing development partition: {train_csv_path}")
    if not os.path.exists(patient_details_path):
        raise FileNotFoundError(f"Missing patient details: {patient_details_path}")

    df_train_dev = pd.read_csv(train_csv_path)
    df_patient = pd.read_csv(patient_details_path)

    # 1. Map each patient to their ground truth label
    patient_label_map = dict(
        zip(df_patient["Patient ID"].astype(str).str.lower(), df_patient["COVID-19 Infection"].astype(str).str.lower())
    )

    # 2. Identify all unique patients in train1.csv
    df_train_dev["patient_id"] = df_train_dev["filename"].apply(extract_patient_id)
    unique_patients = sorted(df_train_dev["patient_id"].unique())

    # Stratify patients by class
    pos_patients = [p for p in unique_patients if "pos" in patient_label_map.get(p, "")]
    neg_patients = [p for p in unique_patients if "neg" in patient_label_map.get(p, "")]

    # 3. Deterministic stratified random split
    rng = np.random.RandomState(seed)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    n_val_pos = int(round(len(pos_patients) * val_ratio))
    n_val_neg = int(round(len(neg_patients) * val_ratio))

    val_patients = sorted(pos_patients[:n_val_pos] + neg_patients[:n_val_neg])
    train_patients = sorted(pos_patients[n_val_pos:] + neg_patients[n_val_neg:])

    # 4. Partition slice rows
    df_train_split = df_train_dev[df_train_dev["patient_id"].isin(train_patients)].copy()
    df_val_split = df_train_dev[df_train_dev["patient_id"].isin(val_patients)].copy()

    # 5. Zero-leakage assertions
    train_pat_set = set(train_patients)
    val_pat_set = set(val_patients)

    assert len(train_pat_set.intersection(val_pat_set)) == 0, "Train and Val share patients!"
    assert len(train_pat_set) + len(val_pat_set) == len(unique_patients), "Patient count mismatch!"

    # Check against held-out test partition
    test_pat_set: Set[str] = set()
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        test_pat_set = set(df_test["filename"].apply(extract_patient_id).unique())
        
        assert len(train_pat_set.intersection(test_pat_set)) == 0, "DATA LEAKAGE: Train shares patients with Test!"
        assert len(val_pat_set.intersection(test_pat_set)) == 0, "DATA LEAKAGE: Val shares patients with Test!"

    # 6. Save manifests
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train_split.csv")
    val_out = os.path.join(output_dir, "val_split.csv")

    df_train_split.to_csv(train_out, index=False)
    df_val_split.to_csv(val_out, index=False)

    summary = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_patients_count": len(train_patients),
        "val_patients_count": len(val_patients),
        "test_patients_count": len(test_pat_set),
        "train_slices_count": len(df_train_split),
        "val_slices_count": len(df_val_split),
        "train_pos_patients": len([p for p in train_patients if "pos" in patient_label_map.get(p, "")]),
        "train_neg_patients": len([p for p in train_patients if "neg" in patient_label_map.get(p, "")]),
        "val_pos_patients": len([p for p in val_patients if "pos" in patient_label_map.get(p, "")]),
        "val_neg_patients": len([p for p in val_patients if "neg" in patient_label_map.get(p, "")]),
        "train_manifest": train_out,
        "val_manifest": val_out,
    }

    print(f"\n[Patient Split Complete — Seed {seed}]")
    print(f"  • Train: {summary['train_patients_count']} patients ({summary['train_pos_patients']} Pos, {summary['train_neg_patients']} Neg) | {summary['train_slices_count']} slices")
    print(f"  • Val:   {summary['val_patients_count']} patients ({summary['val_pos_patients']} Pos, {summary['val_neg_patients']} Neg) | {summary['val_slices_count']} slices")
    print(f"  • Test:  {summary['test_patients_count']} patients (HELD OUT — UNTOUCHED)")
    print(f"  • Overlaps: Train∩Val = 0, Train∩Test = 0, Val∩Test = 0 ✅")
    print(f"  • Saved: {train_out} and {val_out}\n")

    return summary


def validate_split_integrity(filepath: str, partition_name: str) -> None:
    """
    Safeguard function to prevent accidentally passing test1.csv as validation data.
    """
    fname = os.path.basename(filepath).lower()
    if "test" in fname:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE VIOLATION: '{filepath}' contains 'test'. "
            f"The held-out test partition must NEVER be used as a validation set for "
            f"{partition_name}! Use 'val_split.csv' instead."
        )


if __name__ == "__main__":
    create_patient_level_split(seed=42)
