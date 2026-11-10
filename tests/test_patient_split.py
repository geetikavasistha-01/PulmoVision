"""
Automated unit tests for patient-level dataset splitting and anti-leakage safeguards.
"""

import pytest
import os
import pandas as pd
from src.data.patient_split import (
    create_patient_level_split,
    extract_patient_id,
    validate_split_integrity
)


def test_extract_patient_id():
    assert extract_patient_id("137covid_patient100_SR_2_IM00028.tif") == "patient100"
    assert extract_patient_id("normal_patient5_SR_1_IM00003.tif") == "patient5"
    assert extract_patient_id("patient1_IM001.tif") == "patient1"

    with pytest.raises(ValueError):
        extract_patient_id("random_image_without_id.png")


def test_patient_level_split_zero_overlap_and_counts():
    train_csv = "data/metadata/Annotation/CSV/train1.csv"
    patient_csv = "data/metadata/Annotation/CSV/Patient_details.csv"
    test_csv = "data/metadata/Annotation/CSV/test1.csv"

    if not (os.path.exists(train_csv) and os.path.exists(patient_csv)):
        pytest.skip("Metadata CSVs not present on disk")

    summary = create_patient_level_split(
        train_csv_path=train_csv,
        patient_details_path=patient_csv,
        test_csv_path=test_csv,
        seed=42
    )

    # Patient counts
    assert summary["train_patients_count"] == 98
    assert summary["val_patients_count"] == 24
    assert summary["test_patients_count"] == 255

    # Check zero overlap directly from generated manifests
    df_train = pd.read_csv(summary["train_manifest"])
    df_val = pd.read_csv(summary["val_manifest"])
    df_test = pd.read_csv(test_csv)

    set_train = set(df_train["filename"].apply(extract_patient_id))
    set_val = set(df_val["filename"].apply(extract_patient_id))
    set_test = set(df_test["filename"].apply(extract_patient_id))

    assert len(set_train.intersection(set_val)) == 0, "Train and Val share patients!"
    assert len(set_train.intersection(set_test)) == 0, "Train and Test share patients!"
    assert len(set_val.intersection(set_test)) == 0, "Val and Test share patients!"


def test_split_reproducibility():
    train_csv = "data/metadata/Annotation/CSV/train1.csv"
    patient_csv = "data/metadata/Annotation/CSV/Patient_details.csv"
    if not (os.path.exists(train_csv) and os.path.exists(patient_csv)):
        pytest.skip("Metadata CSVs not present on disk")

    res1 = create_patient_level_split(seed=42)
    res2 = create_patient_level_split(seed=42)

    assert res1["train_patients_count"] == res2["train_patients_count"]
    assert res1["val_patients_count"] == res2["val_patients_count"]
    assert res1["train_slices_count"] == res2["train_slices_count"]


def test_test_set_validation_safeguard():
    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE VIOLATION"):
        validate_split_integrity("data/metadata/Annotation/CSV/test1.csv", "Validation")

    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE VIOLATION"):
        validate_split_integrity("test_manifest.csv", "Validation")

    # Valid name should not raise
    validate_split_integrity("val_split.csv", "Validation")

