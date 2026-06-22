"""Dataset PyTorch : (image, sexe) -> âge osseux (mois).

Lit splits.csv généré par preprocess.py. Si data/processed/img/ existe (cache),
les images pré-rendues sont utilisées ; sinon lecture+prétraitement à la volée.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from preprocess import preprocess_image


class BoneAgeDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        df = pd.read_csv(C.DATA_PROC / "splits.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment
        self.cache_dir = C.DATA_PROC / "img"
        self.use_cache = self.cache_dir.exists()

    def __len__(self):
        return len(self.df)

    def _load(self, _id: str) -> np.ndarray:
        if self.use_cache:
            p = self.cache_dir / f"{_id}.png"
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:  # repli si image absente du cache
                img = preprocess_image(C.TRAIN_IMG_DIR / f"{_id}.png", C.IMG_SIZE, C.USE_CLAHE)
        else:
            img = preprocess_image(C.TRAIN_IMG_DIR / f"{_id}.png", C.IMG_SIZE, C.USE_CLAHE)
        return img

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # flip horizontal
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        # petite rotation
        if np.random.rand() < 0.5:
            ang = np.random.uniform(-12, 12)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)
        # jitter d'intensité
        if np.random.rand() < 0.5:
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load(str(row["id"]))
        if self.augment:
            img = self._augment(img)

        # -> tensor 3 canaux (réplication du gris), normalisation ImageNet
        x = img.astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=0)  # (3,H,W)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = (x - mean) / std

        sex = np.float32(row["male"])                      # 1=garçon, 0=fille
        age = np.float32((row["age"] - C.AGE_MEAN) / C.AGE_STD)  # cible standardisée

        return (
            torch.from_numpy(x),
            torch.tensor([sex], dtype=torch.float32),
            torch.tensor([age], dtype=torch.float32),
        )
