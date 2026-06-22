"""Outil d'EXAMEN — évalue le modèle entraîné et mesure sa précision.

Ce que fait ce script, en clair :
  1. Il recharge le meilleur modèle obtenu pendant l'entraînement (runs/best.pt).
  2. Il lui fait deviner l'âge osseux sur des radios qu'il n'a pas servi à l'entraîner.
  3. Il compare ses réponses à la vérité et calcule des "notes" :
       - MAE  : erreur moyenne en mois (la note principale du challenge RSNA).
       - RMSE : pénalise davantage les grosses erreurs.
       - % ±12 mois : part des prédictions à moins d'un an de la vérité.
  4. Il enregistre toutes les prédictions dans un fichier (predictions.csv)
     et trace deux graphiques faciles à lire (runs/eval/).

Usage :
    python src/predict.py                 # évalue sur le split "val"
    python src/predict.py --split val     # idem
    python src/predict.py --ckpt runs/best.pt
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import BoneAgeDataset
from model import BoneAgeModel


@torch.no_grad()
def run_predictions(model, loader, device, age_mean, age_std):
    """Renvoie deux listes alignées : âges prédits et âges réels (en mois)."""
    model.eval()
    preds, trues = [], []
    for img, sex, age in loader:
        img, sex = img.to(device), sex.to(device)
        with torch.autocast(device_type=device.type, enabled=C.AMP):
            out = model(img, sex)
            out_flip = model(torch.flip(img, dims=[3]), sex)  # TTA: image miroir
        out = (out.float() + out_flip.float()) / 2
        # on repasse de la valeur standardisée vers des mois
        preds.append(out.float().cpu().squeeze(1) * age_std + age_mean)
        trues.append(age.squeeze(1) * age_std + age_mean)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def compute_metrics(pred, true):
    err = pred - true
    abs_err = np.abs(err)
    return {
        "n": int(len(true)),
        "MAE_mois": float(abs_err.mean()),
        "RMSE_mois": float(np.sqrt((err ** 2).mean())),
        "pct_dans_12_mois": float((abs_err <= 12).mean() * 100),
        "pct_dans_24_mois": float((abs_err <= 24).mean() * 100),
        "biais_moyen_mois": float(err.mean()),  # >0 = surestime, <0 = sous-estime
    }


def make_plots(pred, true, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prédit vs réel : les points doivent suivre la diagonale
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=8, alpha=0.4)
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="prédiction parfaite")
    plt.xlabel("Âge réel (mois)"); plt.ylabel("Âge prédit (mois)")
    plt.title("Prédit vs réel"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true.png", dpi=120); plt.close()

    # 2) Bland-Altman : visualise l'erreur et un éventuel biais systématique
    mean_ax = (pred + true) / 2
    diff = pred - true
    md, sd = diff.mean(), diff.std()
    plt.figure(figsize=(7, 5))
    plt.scatter(mean_ax, diff, s=8, alpha=0.4)
    plt.axhline(md, color="k", label=f"biais moyen = {md:.1f}")
    plt.axhline(md + 1.96 * sd, color="grey", ls="--", label="±1,96 écart-type")
    plt.axhline(md - 1.96 * sd, color="grey", ls="--")
    plt.xlabel("Moyenne (prédit, réel) en mois"); plt.ylabel("Écart prédit - réel (mois)")
    plt.title("Bland-Altman"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "bland_altman.png", dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", help="val (par défaut) ou train")
    ap.add_argument("--ckpt", default=str(C.RUNS / "best.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {ckpt_path}\n"
            "Lance d'abord l'entraînement (python src/train.py)."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    age_mean = ckpt.get("age_mean", C.AGE_MEAN)
    age_std = ckpt.get("age_std", C.AGE_STD)

    model = BoneAgeModel(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Modèle chargé (epoch {ckpt.get('epoch','?')}, "
          f"MAE val enregistrée = {ckpt.get('val_mae','?')}).")

    ds = BoneAgeDataset(args.split, augment=False)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False,
                    num_workers=C.NUM_WORKERS, pin_memory=True)
    print(f"Évaluation sur le split '{args.split}' : {len(ds)} radios")

    pred, true = run_predictions(model, dl, device, age_mean, age_std)
    metrics = compute_metrics(pred, true)

    print("\n========== RÉSULTATS ==========")
    print(f"Nombre de radios      : {metrics['n']}")
    print(f"MAE (erreur moyenne)  : {metrics['MAE_mois']:.2f} mois")
    print(f"RMSE                  : {metrics['RMSE_mois']:.2f} mois")
    print(f"% à moins d'1 an      : {metrics['pct_dans_12_mois']:.1f} %")
    print(f"% à moins de 2 ans    : {metrics['pct_dans_24_mois']:.1f} %")
    print(f"Biais moyen           : {metrics['biais_moyen_mois']:+.2f} mois "
          "(positif = surestime, négatif = sous-estime)")
    print("Repère : gagnant RSNA 2017 ≈ 4,3 mois ; radiologues ≈ 5,8 mois.")
    print("===============================\n")

    out_dir = C.RUNS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # fichier détaillé des prédictions
    df_ids = pd.read_csv(C.DATA_PROC / "splits.csv")
    df_ids = df_ids[df_ids["split"] == args.split].reset_index(drop=True)
    out_df = pd.DataFrame({
        "id": df_ids["id"],
        "sexe": df_ids["male"].map({1: "garcon", 0: "fille"}),
        "age_reel_mois": np.round(true, 1),
        "age_predit_mois": np.round(pred, 1),
        "erreur_mois": np.round(pred - true, 1),
    })
    out_df.to_csv(out_dir / "predictions.csv", index=False)

    # métriques globales + par sexe
    rows = [{"groupe": "tous", **metrics}]
    for label, name in [(1, "garcons"), (0, "filles")]:
        mask = (df_ids["male"] == label).values
        if mask.any():
            rows.append({"groupe": name, **compute_metrics(pred[mask], true[mask])})
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    make_plots(pred, true, out_dir)
    print(f"Fichiers écrits dans {out_dir} :")
    print("  - predictions.csv  (toutes les prédictions, radio par radio)")
    print("  - metrics.csv      (notes globales + par sexe)")
    print("  - pred_vs_true.png et bland_altman.png  (graphiques)")


if __name__ == "__main__":
    main()
