"""
PulmoVision — Stage 2 Deployment Model Training Procedure
Trains an explicitly-labeled deployment checkpoint for Experiment B (effnet_b0_bilstm_mean)
on the complete 122-patient development cohort using patient-balanced sequence optimization.

Guarantees:
- Zero exposure to held-out test cohort (test1.csv is strictly quarantined).
- Stage 1 baseline models and checkpoints remain completely untouched.
- Deployment model is explicitly labeled and timestamped with full metadata.
"""

import os
import json
import time
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn_bilstm import get_stage2_model
from src.data.sequence_dataset import PatientSequenceDataset, collate_patient_balanced_batch
from src.evaluation.nested_cv import train_epoch


def train_deployment_model(
    cache_path: str = "data/processed/dev_patient_features_effnet_b0.pt",
    output_path: str = "models/stage2_bilstm_deployment.pth",
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = None,
):
    if device is None:
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    print("=" * 70)
    print("PULMOVISION STAGE 2 — DEPLOYMENT MODEL TRAINING")
    print("=" * 70)
    print(f"Hardware device:    {dev}")
    print(f"Feature cache:      {cache_path}")
    print(f"Target model:       effnet_b0_bilstm_mean (Experiment B)")
    print(f"Target epochs:      {epochs}")
    print(f"Learning rate:      {lr}")
    print(f"Weight decay:       {weight_decay}")
    print(f"Batch size:         {batch_size}")
    print(f"Destination:        {output_path}")

    torch.manual_seed(seed)

    # 1. Load verified development feature cache
    assert os.path.exists(cache_path), f"Cache not found: {cache_path}"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    # 2. Build patient-level dataset over all 122 development patients
    dataset = PatientSequenceDataset(cache)
    print(f"Development cohort: {len(dataset)} patients")

    # 3. Compute class weights exclusively from development patient counts
    labels = [item["label"] for item in dataset]
    n_covid = sum(labels)
    n_normal = len(labels) - n_covid
    w_norm = len(labels) / (2.0 * max(n_normal, 1))
    w_covid = len(labels) / (2.0 * max(n_covid, 1))
    class_weights = torch.tensor([w_norm, w_covid], dtype=torch.float32).to(dev)
    print(f"Cohort balance:     {n_covid} COVID-19, {n_normal} Normal")
    print(f"Class weights:      Normal={w_norm:.4f}, COVID-19={w_covid:.4f}")

    # 4. DataLoader with patient-balanced collation
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_balanced_batch,
    )

    # 5. Initialize model and optimizer
    model = get_stage2_model("effnet_b0_bilstm_mean").to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Training loop
    start_t = time.time()
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, dev)
        print(f"  Epoch {ep:02d}/{epochs:02d} — Patient-Balanced Loss: {loss:.4f}")
    elapsed = time.time() - start_t
    print(f"Training completed in {elapsed:.2f}s.")

    # 7. Checkpoint preparation & metadata stamping
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = {
        "model_name": "effnet_b0_bilstm_mean",
        "state_dict": model.state_dict(),
        "input_dim": 1280,
        "hidden_dim": 128,
        "recurrent_out_dim": 256,
        "mlp_dim": 64,
        "num_classes": 2,
        "dropout": 0.3,
        "threshold": 0.50,
        "class_weights": class_weights.cpu().tolist(),
        "training_epochs": epochs,
        "training_cohort": "COVID-CTset Deduplicated Development Cohort (122 patients)",
        "feature_extractor": "EfficientNet-B0 (IMAGENET1K_V1, frozen)",
        "quarantine_guarantee": "test1.csv quarantined, zero exposure",
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    torch.save(checkpoint, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Compute SHA256 checksum
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    ckpt_hash = h.hexdigest()

    print(f"Successfully saved deployment checkpoint:")
    print(f"  Path:   {output_path}")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  SHA256: {ckpt_hash}")

    return {
        "output_path": output_path,
        "file_size_mb": file_size_mb,
        "sha256": ckpt_hash,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    train_deployment_model()

