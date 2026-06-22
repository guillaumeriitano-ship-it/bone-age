"""J3-6 — Entraînement baseline régression.

- Perte Huber (SmoothL1) sur l'âge standardisé.
- MAE de validation reportée en MOIS (métrique RSNA).
- Mixed precision (AMP), cosine LR, early stopping, sauvegarde du meilleur modèle.

Usage:
    python src/train.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from dataset import BoneAgeDataset
from model import BoneAgeModel, denormalize_age


def recompute_age_stats():
    """Met à jour AGE_MEAN/AGE_STD à partir du split train réel."""
    sp = C.DATA_PROC / "splits.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        tr = df[df["split"] == "train"]["age"].astype(float)
        C.AGE_MEAN = float(tr.mean())
        C.AGE_STD = float(tr.std())
        print(f"Stats âge (train) : mean={C.AGE_MEAN:.1f}  std={C.AGE_STD:.1f}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    abs_err = []
    for img, sex, age in loader:
        img, sex = img.to(device), sex.to(device)
        with torch.autocast(device_type=device.type, enabled=C.AMP):
            pred = model(img, sex)
        pred_m = denormalize_age(pred.float().cpu()).squeeze(1)
        true_m = denormalize_age(age).squeeze(1)
        abs_err.append((pred_m - true_m).abs())
    return float(torch.cat(abs_err).mean())  # MAE en mois


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    C.RUNS.mkdir(parents=True, exist_ok=True)
    recompute_age_stats()

    train_ds = BoneAgeDataset("train", augment=True)
    val_ds = BoneAgeDataset("val", augment=False)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)

    model = BoneAgeModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)
    # Huber en unités standardisées (delta converti)
    loss_fn = nn.SmoothL1Loss(beta=C.HUBER_DELTA / C.AGE_STD)
    scaler = torch.cuda.amp.GradScaler(enabled=C.AMP)

    best_mae = float("inf")
    patience = 0
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for img, sex, age in tqdm(train_dl, desc=f"epoch {epoch}/{C.EPOCHS}"):
            img, sex, age = img.to(device), sex.to(device), age.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=C.AMP):
                pred = model(img, sex)
                loss = loss_fn(pred, age)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * img.size(0)
        sched.step()

        train_loss = running / len(train_ds)
        val_mae = evaluate(model, val_dl, device)
        dt = time.time() - t0
        print(f"[{epoch:02d}] train_loss={train_loss:.4f}  val_MAE={val_mae:.2f} mois  ({dt:.0f}s)")

        if val_mae < best_mae:
            best_mae = val_mae
            patience = 0
            torch.save({"model": model.state_dict(),
                        "age_mean": C.AGE_MEAN, "age_std": C.AGE_STD,
                        "epoch": epoch, "val_mae": val_mae},
                       C.RUNS / "best.pt")
            print(f"  ✓ nouveau meilleur modèle (MAE={best_mae:.2f}) sauvegardé")
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                print(f"Early stopping (pas d'amélioration depuis {patience} epochs).")
                break

    print(f"\nMeilleure MAE validation : {best_mae:.2f} mois  -> {C.RUNS / 'best.pt'}")


if __name__ == "__main__":
    main()
