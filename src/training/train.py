"""
PulmoVision — Main Training Pipeline
Implements class-weighted optimization, early stopping, and checkpointing
for 2.5D deep learning models.
"""

import os
import argparse
import yaml
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.model_factory import get_model
from src.training.validate import evaluate_epoch
from src.training.checkpoint import save_checkpoint


def load_config(config_path: str = "configs/training.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict[str, Any]:
    """
    Main training execution function.
    """
    # 1. Device selection: Apple MPS -> CUDA -> CPU
    device_name = config["training"].get("device", "mps")
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n[Training] Using compute device: {device}")

    # 2. Model instantiation
    m_cfg = config["model"]
    d_cfg = config["data"]
    model = get_model(
        model_name=m_cfg["name"],
        in_channels=d_cfg["in_channels"],
        num_classes=m_cfg["num_classes"],
        pretrained=m_cfg["pretrained"],
        dropout=m_cfg["dropout"],
    ).to(device)

    # 3. Class-weighted loss for 1:3 imbalanced dataset
    weights = torch.tensor(config["training"]["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 4. Optimizer and LR Scheduler
    t_cfg = config["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["learning_rate"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    epochs = int(t_cfg["epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Tracking and Early Stopping
    best_val_f1 = -1.0
    patience = int(t_cfg["early_stopping_patience"])
    patience_counter = 0
    save_path = os.path.join(config["checkpoint"]["save_dir"], config["checkpoint"]["filename"])

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print("=" * 70)
    print(f"PULMOVISION TRAINING: {m_cfg['name'].upper()} (2.5D, 3 Channels)")
    print("=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1 (Macro)':^14}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)
            n_train += batch_x.size(0)

        epoch_train_loss = total_train_loss / max(1, n_train)
        scheduler.step()

        # Validation Phase
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_acc"]
        val_f1 = val_metrics["val_f1"]

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {val_loss:^10.4f} | {val_acc:^9.1%} | {val_f1:^14.4f}")

        # Checkpoint if best Macro F1 achieved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                filepath=save_path,
                model_name=m_cfg["name"],
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered after {patience} epochs without improvement.")
                break

    print("=" * 70)
    print(f"Training Complete! Best Validation Macro F1: {best_val_f1:.4f}")
    return history


def create_real_dataloaders(
    data_dir: str = "data/raw",
    metadata_dir: str = "data/metadata/Annotation/CSV",
    train_manifest: str = "train_split.csv",
    val_manifest: str = "val_split.csv",
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs genuine 2.5D DataLoaders from extracted COVID-CTset files and approved split manifests.
    Enforces that test1.csv is NEVER used as a validation partition.
    """
    import glob
    import pandas as pd
    from src.data.slice_stacking import CTSliceStacker
    from src.data.dataloader import create_dataloader
    from src.data.preprocessing import CTPreprocessor
    from src.data.patient_split import validate_split_integrity

    train_csv = os.path.join(metadata_dir, train_manifest)
    val_csv = os.path.join(metadata_dir, val_manifest)

    # 1. Critical Anti-Leakage Safeguard
    validate_split_integrity(val_csv, "Validation Partition")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        raise FileNotFoundError(
            f"Split manifests not found: '{train_csv}' or '{val_csv}'. "
            f"Please run 'python src/data/patient_split.py' first."
        )

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Locate image files on disk
    train_paths = [os.path.join(data_dir, "TrainValidation", f) for f in df_train["filename"] if os.path.exists(os.path.join(data_dir, "TrainValidation", f))]
    val_paths = [os.path.join(data_dir, "TrainValidation", f) for f in df_val["filename"] if os.path.exists(os.path.join(data_dir, "TrainValidation", f))]

    print(f"[DataLoader] Found {len(train_paths):,} train slice files, {len(val_paths):,} val slice files on disk.")

    train_triplets = CTSliceStacker.generate_triplets(train_paths)
    val_triplets = CTSliceStacker.generate_triplets(val_paths)

    train_prep = CTPreprocessor(target_size=(224, 224), is_training=True)
    val_prep = CTPreprocessor(target_size=(224, 224), is_training=False)

    train_loader = create_dataloader(train_triplets, preprocessor=train_prep, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_triplets, preprocessor=val_prep, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_synthetic_dataloaders(batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
    """
    Creates mock 2.5D [3, 224, 224] dataloaders for verification & pipeline dry-runs.
    """
    torch.manual_seed(42)
    X_train = torch.randn(64, 3, 224, 224)
    y_train = torch.tensor([0] * 48 + [1] * 16)  # 3:1 ratio
    X_val = torch.randn(32, 3, 224, 224)
    y_val = torch.tensor([0] * 24 + [1] * 8)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulmoVision Model Training")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic verification batches")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    if args.dry_run or not os.path.exists(os.path.join(args.data_dir, "TrainValidation")):
        if not args.dry_run:
            print(f"[Warning] '{args.data_dir}/TrainValidation' not found. Falling back to dry-run synthetic mode.")
        t_loader, v_loader = create_synthetic_dataloaders(batch_size=cfg["data"]["batch_size"])
    else:
        t_loader, v_loader = create_real_dataloaders(data_dir=args.data_dir, batch_size=cfg["data"]["batch_size"])
        
    train_model(cfg, t_loader, v_loader)

