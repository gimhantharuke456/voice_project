"""
Step 6 — CDAE Trainer
Trains the model from Step 5 on spectrograms from Step 4.
Saves best weights to models/cdae_best.pth and a loss curve to outputs/plots/.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from step4_preprocessor import SpectrogramDataset
from step5_model import CDAE

# Hyperparameters
EPOCHS      = 30
BATCH_SIZE  = 16
LR          = 1e-3
NUM_WORKERS = 0

MODEL_PATH = "models/cdae_best.pth"
PLOT_PATH  = "outputs/plots/loss_curve.png"

BOLD  = "\033[1m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; RESET = "\033[0m"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out  = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out  = model(noisy)
            loss = criterion(out, clean)
            total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def save_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.plot(range(1, len(val_losses) + 1),   val_losses,   label="Val loss", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("CDAE Training Loss Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 6: Trainer ==={RESET}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {device}")
if device.type == "cpu":
    print(f"{YELLOW}  (No GPU — training on CPU. Use Google Colab for faster runs.){RESET}")

train_ds = SpectrogramDataset(split="train")
val_ds   = SpectrogramDataset(split="val")

if len(train_ds) == 0:
    print("No training data found. Run Steps 3 and 4 first.")
    sys.exit(1)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train      : {len(train_ds):,} samples  ({len(train_loader)} batches)")
print(f"Val        : {len(val_ds):,} samples  ({len(val_loader)} batches)")
print(f"Epochs     : {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LR}\n")

model     = CDAE().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

start_epoch  = 0
best_val     = float("inf")
train_losses = []
val_losses   = []

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch  = ckpt["epoch"]
        best_val     = ckpt["best_val"]
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
        print(f"Resumed from epoch {start_epoch}  (best val loss: {best_val:.6f})\n")

print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'LR':>10}  {'Time':>8}  {'Status'}")
print("-" * 68)

for epoch in range(start_epoch + 1, EPOCHS + 1):
    t0         = time.time()
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss   = validate(model, val_loader, criterion, device)
    elapsed    = time.time() - t0

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    status     = ""

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "best_val":     best_val,
            "train_losses": train_losses,
            "val_losses":   val_losses,
        }, MODEL_PATH)
        status = "saved"

    print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}  {elapsed:>7.1f}s  {status}")

save_loss_curve(train_losses, val_losses)

print(f"\n{'='*55}")
print(f"{BOLD}Summary{RESET}")
print(f"  Best val loss : {best_val:.6f}")
print(f"  Model saved   : {MODEL_PATH}")
print(f"  Loss curve    : {PLOT_PATH}")
print(f"{'='*55}")
print(f"\n{GREEN}Training complete. Proceed to Step 7.{RESET}\n")
