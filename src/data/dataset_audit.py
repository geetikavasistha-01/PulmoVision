"""
PulmoVision Dataset Audit Script
Inspects COVID-CTset metadata, patient identifiers, slice ordering,
image resolution, bit depth, and class balance without downloading the full dataset.
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image

def run_audit():
    meta_path = "data/metadata/Annotation/CSV/Patient_details.csv"
    train_path = "data/metadata/Annotation/CSV/train1.csv"
    test_path = "data/metadata/Annotation/CSV/test1.csv"
    sample_img_path = "data/interim/sample_audit/TrainValidation/137covid_patient100_SR_2_IM00028.tif"

    print("=" * 60)
    print("PULMOVISION — COVID-CTSET METADATA AUDIT")
    print("=" * 60)

    # 1. Patient Details CSV
    df_patient = pd.read_csv(meta_path)
    print(f"\n[1] Patient_details.csv Shape: {df_patient.shape}")
    print("Columns:", list(df_patient.columns))
    print("\nFirst 3 rows:")
    print(df_patient.head(3))

    # Unique patients and infection column
    inf_col = [c for c in df_patient.columns if 'infection' in c.lower() or 'covid' in c.lower() or 'diagnosis' in c.lower()][0]
    print(f"\nInfection Column Identified: '{inf_col}'")
    print("Patient-level Class Distribution:")
    print(df_patient[inf_col].value_counts(dropna=False))

    # 2. train1.csv and test1.csv inspection
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"\n[2] Fold 1 Split Analysis:")
    print(f"train1.csv shape: {df_train.shape}, columns: {list(df_train.columns)}")
    print(f"test1.csv shape: {df_test.shape}, columns: {list(df_test.columns)}")
    print("\ntrain1 sample:")
    print(df_train.head(3))

    # Extract patient IDs from image filenames in train and test
    def extract_patient_id(filename):
        # Format example: 137covid_patient100_SR_2_IM00028.tif or normal_patient...
        match = re.search(r'(patient\d+)', str(filename), re.IGNORECASE)
        return match.group(1).lower() if match else 'unknown'

    img_col = df_train.columns[0]
    train_patients = set(df_train[img_col].apply(extract_patient_id))
    test_patients = set(df_test[img_col].apply(extract_patient_id))
    overlap = train_patients.intersection(test_patients)

    print(f"\n[3] Patient Leakage Check in Fold 1:")
    print(f"Train unique patients: {len(train_patients)}")
    print(f"Test unique patients: {len(test_patients)}")
    print(f"Patient overlap between train and test: {len(overlap)}")
    if len(overlap) > 0:
        print(f"⚠️ LEAKAGE DETECTED: {list(overlap)[:5]}")
    else:
        print("✅ NO PATIENT LEAKAGE in author's official fold 1!")

    # 3. Image Properties
    print("\n[4] Physical Sample Image Inspection:")
    if os.path.exists(sample_img_path):
        img = Image.open(sample_img_path)
        arr = np.array(img)
        print(f"File: {os.path.basename(sample_img_path)}")
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        print(f"Dimensions (H, W): {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Pixel Value Range: [{arr.min()}, {arr.max()}]")
        print(f"Mean: {arr.mean():.2f}, Std: {arr.std():.2f}")
    else:
        print("Sample image not found on disk.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_audit()

