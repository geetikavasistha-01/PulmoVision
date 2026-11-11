"""
PulmoVision — Full Feature Extraction Execution Script
Extracts 1280D ImageNet-1K EfficientNet-B0 embeddings for the 122 development patients
across all 239 CT series (3,736 total slices) without touching test1.csv.
"""
import time
import os
import torch
from src.features.feature_extractor import PretrainedFeatureExtractor

def main():
    start_time = time.time()
    print("Initializing PretrainedFeatureExtractor on MPS...")
    extractor = PretrainedFeatureExtractor()
    print(f"Device: {extractor.device}")
    print(f"Backbone: {extractor.WEIGHTS_ID}")

    manifest_path = "data/metadata/Annotation/CSV/train1.csv"
    data_dir = "data/raw/TrainValidation"
    output_path = "data/processed/dev_patient_features_effnet_b0.pt"

    print(f"Beginning full extraction from {manifest_path}...")
    cache = extractor.extract_and_build_patient_manifest(
        manifest_csv=manifest_path,
        data_dir=data_dir,
        output_path=output_path,
        batch_size=32
    )

    elapsed = time.time() - start_time
    total_slices = cache["metadata"]["total_slices"]
    throughput = total_slices / elapsed if elapsed > 0 else 0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("=" * 60)
    print("EXTRACTION COMPLETED SUCCESSFULLY")
    print(f"Total Patients:      {len(cache['patients'])}")
    print(f"Total Series:        {len(cache['series'])}")
    print(f"Total Slices:        {total_slices}")
    print(f"Elapsed Time:        {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"Throughput:          {throughput:.2f} slices/sec")
    print(f"Output File:         {output_path}")
    print(f"File Size:           {file_size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()

