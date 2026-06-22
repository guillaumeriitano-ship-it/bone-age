"""J2 — Prétraitement + création des splits.

- Lit le CSV d'entraînement, normalise la colonne sexe.
- Crée un split train/val reproductible (stratifié par tranche d'âge) si pas de val labellisée.
- Écrit data/processed/splits.csv  (colonnes: id, age, male, split)
- (Option) pré-rend les images redimensionnées + CLAHE dans data/processed/img/
  pour accélérer l'entraînement (sinon le Dataset lit les PNG bruts à la volée).

Usage:
    python src/preprocess.py            # crée splits.csv (lecture à la volée)
    python src/preprocess.py --cache    # + pré-rend les images
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(C.TRAIN_CSV)
    male = df[C.COL_MALE]
    if male.dtype == object:
        male = male.astype(str).str.lower().isin(["true", "1", "m", "male"])
    df = pd.DataFrame({
        "id": df[C.COL_ID].astype(str),
        "age": df[C.COL_AGE].astype(float),
        "male": male.astype(int),
    })
    return df


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    # Stratification par décile d'âge pour équilibrer train/val
    bins = pd.qcut(df["age"], q=10, labels=False, duplicates="drop")
    train_idx, val_idx = train_test_split(
        df.index, test_size=C.VAL_FRACTION, random_state=C.SEED, stratify=bins
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    return df


def preprocess_image(path: Path, size: int, use_clahe: bool) -> np.ndarray:
    """Lit une radio en niveaux de gris, normalise, CLAHE optionnel, resize carré."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Image illisible : {path}")
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    # padding carré pour conserver le ratio
    h, w = img.shape
    m = max(h, w)
    canvas = np.zeros((m, m), dtype=img.dtype)
    canvas[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="pré-rendre les images sur disque")
    args = ap.parse_args()

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    df = make_splits(df)
    splits_path = C.DATA_PROC / "splits.csv"
    df.to_csv(splits_path, index=False)
    print(f"splits.csv écrit : {splits_path}  "
          f"(train={ (df.split=='train').sum() }, val={ (df.split=='val').sum() })")

    if args.cache:
        out_dir = C.DATA_PROC / "img"
        out_dir.mkdir(parents=True, exist_ok=True)
        n, errs = 0, 0
        for _id in df["id"]:
            src = C.TRAIN_IMG_DIR / f"{_id}.png"
            try:
                im = preprocess_image(src, C.IMG_SIZE, C.USE_CLAHE)
                cv2.imwrite(str(out_dir / f"{_id}.png"), im)
                n += 1
            except Exception as e:  # noqa: BLE001
                errs += 1
                if errs <= 5:
                    print(f"  warn: {e}")
            if n % 1000 == 0 and n:
                print(f"  {n} images traitées…")
        print(f"Cache terminé : {n} images écrites dans {out_dir} ({errs} erreurs)")


if __name__ == "__main__":
    main()
