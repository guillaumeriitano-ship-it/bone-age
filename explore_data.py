"""J1 — Exploration des données (EDA) + contrôle qualité.

Usage:
    python src/explore_data.py
Génère un résumé console et des figures dans runs/eda/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import config as C


def main():
    out = C.RUNS / "eda"
    out.mkdir(parents=True, exist_ok=True)

    if not C.TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {C.TRAIN_CSV}\n"
            "Télécharge les données (voir README) puis ajuste src/config.py."
        )

    df = pd.read_csv(C.TRAIN_CSV)
    print(f"Lignes CSV : {len(df)}")
    print(f"Colonnes  : {list(df.columns)}")

    # Normalise la colonne sexe en booléen
    male = df[C.COL_MALE]
    if male.dtype == object:
        male = male.astype(str).str.lower().isin(["true", "1", "m", "male"])
    df["_male"] = male.astype(int)

    age = df[C.COL_AGE].astype(float)
    print("\n--- Âge osseux (mois) ---")
    print(age.describe())
    print(f"\nRépartition sexe : garçons={df['_male'].sum()}  filles={(1 - df['_male']).sum()}")

    # Vérifie la présence physique des images (échantillon)
    missing = 0
    sample = df.head(500)
    for _id in sample[C.COL_ID]:
        p = C.TRAIN_IMG_DIR / f"{_id}.png"
        if not p.exists():
            missing += 1
    print(f"\nImages manquantes sur 500 testées : {missing}")
    if missing:
        print("  -> vérifie TRAIN_IMG_DIR et l'extension des fichiers dans config.py")

    # Dimensions d'un échantillon
    try:
        first = C.TRAIN_IMG_DIR / f"{df[C.COL_ID].iloc[0]}.png"
        with Image.open(first) as im:
            print(f"Exemple image {first.name} : taille={im.size} mode={im.mode}")
    except Exception as e:  # noqa: BLE001
        print(f"Lecture image échantillon impossible : {e}")

    # Figures
    plt.figure(figsize=(7, 4))
    age.hist(bins=40)
    plt.xlabel("Âge osseux (mois)"); plt.ylabel("Effectif"); plt.title("Distribution âge osseux")
    plt.tight_layout(); plt.savefig(out / "age_hist.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    for label, g in df.groupby("_male"):
        g[C.COL_AGE].astype(float).hist(bins=40, alpha=0.5,
                                        label="Garçons" if label else "Filles")
    plt.xlabel("Âge osseux (mois)"); plt.ylabel("Effectif")
    plt.legend(); plt.title("Âge osseux par sexe")
    plt.tight_layout(); plt.savefig(out / "age_by_sex.png", dpi=120); plt.close()

    print(f"\nFigures écrites dans {out}")


if __name__ == "__main__":
    main()
