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


def crop_to_hand(img):
    """Recadre l'image sur la main (Phase 3).

    Sépare la main (claire) du fond (sombre) par seuillage d'Otsu, nettoie le
    masque par morphologie, garde la plus grande composante connexe, puis
    recadre sur sa boîte englobante. Robuste et sans entraînement supplémentaire.
    En cas de doute, renvoie l'image d'origine (pas de recadrage).
    """
    if img is None or img.size == 0:
        return img
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return img
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[largest, :4]
    # garde-fou : la main doit occuper une part raisonnable de l'image
    if w * h < 0.05 * img.shape[0] * img.shape[1]:
        return img
    m = int(0.04 * max(img.shape))
    y0, y1 = max(0, y - m), min(img.shape[0], y + h + m)
    x0, x1 = max(0, x - m), min(img.shape[1], x + w + m)
    crop = img[y0:y1, x0:x1]
    return crop if crop.size else img


def preprocess_image(path: Path, size: int, use_clahe: bool) -> np.ndarray:
    """Lit une radio en niveaux de gris, normalise, CLAHE optionnel, resize carré."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Image illisible : {path}")
    if getattr(C, "USE_CROP", False):
        img = crop_to_hand(img)
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
