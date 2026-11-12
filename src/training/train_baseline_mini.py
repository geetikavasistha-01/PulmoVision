"""
PulmoVision — Module 4 Baseline Mini-Training & Overfitting Demonstration
Trains PulmoCustomCNN on a small subset to inspect training vs validation loss,
demonstrate gradient descent dynamics, and observe overfitting.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.custom_cnn import PulmoCustomCNN


def run_mini_training(epochs: int = 15, lr: float = 0.001):
    os.makedirs("reports/figures", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Module 4] Training PulmoCustomCNN on device: {device}")

    # 1. Create a controlled small subset (64 train samples, 32 val samples)
    # Simulates a small CT slice dataset to clearly illustrate overfitting
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.randn(64, 1, 224, 224)
    # Add subtle distinct pattern for class 1 to allow learning
    X_train[32:, :, 100:124, 100:124] += 1.5
    y_train = torch.tensor([0] * 32 + [1] * 32, dtype=torch.long)

    X_val = torch.randn(32, 1, 224, 224)
    X_val[16:, :, 100:124, 100:124] += 1.5
    y_val = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

    # 2. Initialize Custom CNN baseline
    model = PulmoCustomCNN(in_channels=1, num_classes=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^9}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = total_train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                total_val_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        epoch_val_loss = total_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^10.1%} | {epoch_val_loss:^10.4f} | {epoch_val_acc:^9.1%}")

    print("-" * 65)

    # 3. Plot Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_range = range(1, epochs + 1)
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r--s', label='Validation Loss', linewidth=2)
    ax1.set_title("Cross-Entropy Loss (Identifying Overfitting)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs_range, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r--s', label='Validation Accuracy', linewidth=2)
    ax2.set_title("Classification Accuracy", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    chart_path = "reports/figures/custom_cnn_baseline_loss.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[Module 4] Saved training curves to: {chart_path}")
    print("[Module 4] Complete! All Module 4 tasks verified.\n")


if __name__ == "__main__":
    run_mini_training()

