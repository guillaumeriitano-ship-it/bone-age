# Bone Age — Phase 0/1 (Sprint 15 jours)

Pipeline d'estimation automatique de l'âge osseux (radio main/poignet) sur **RSNA Bone Age**.
Cible Phase 1 : baseline **EfficientNetV2-S + fusion du sexe**, régression, MAE < 8 mois sur validation.
Exécution prévue sur **GPU cloud (RunPod ou équivalent)**.

---

## Planning compressé 15 jours

| Jours | Bloc | Livrable |
|---|---|---|
| **J1** | Setup RunPod + téléchargement données + EDA | Données vérifiées, distributions âge×sexe |
| **J2** | Prétraitement (resize, normalisation, CLAHE optionnel) + splits | `train/val` figés, cache images |
| **J3-4** | Baseline EfficientNetV2-S + fusion sexe, régression | 1ʳᵉ MAE de validation (objectif < 8 mois) |
| **J5-6** | Augmentation + tuning (LR, image size, Huber) | MAE 6-7 mois |
| **J7-8** | Segmentation main (U-Net) + CLAHE + recadrage centré | MAE 5,5-6,5 mois |
| **J9-10** | Attention / régions critiques | MAE ~5 mois |
| **J11-12** | Ensemble (multi-seeds / ConvNeXt) + TTA + calibration | MAE 4,3-4,8 mois (test RSNA) |
| **J13** | Validation externe + analyse d'erreur par sexe/âge | Rapport robustesse |
| **J14** | Interprétabilité (Grad-CAM) + intervalle de confiance | Cartes d'attention |
| **J15** | Gel du modèle + rapport final + export | Modèle + README résultats |

> Ce dépôt couvre **J1-J6** (Phase 0/1). Les modules attention/ensemble (J7+) s'ajouteront ensuite.

---

## 1. Setup RunPod (J1)

1. Créer un pod **PyTorch** (image `runpod/pytorch:2.x-cuda12.x`), GPU ≥ 16 Go (RTX A5000 / A40 / L40 suffisent).
2. Attacher un **Network Volume** (≥ 50 Go) monté sur `/workspace` pour persister données + checkpoints.
3. Dans le terminal du pod :

```bash
cd /workspace
git clone <ton-repo> bone-age   # ou upload ce dossier
cd bone-age
pip install -r requirements.txt
```

## 2. Téléchargement des données (J1)

Tu as accès à **Stanford AIMI**. Deux voies :

**Option A — Stanford AIMI (source officielle, recommandée)**
Connecte-toi sur https://aimi.stanford.edu/rsna-bone-age, accepte la licence (recherche/non-commercial), récupère les liens et télécharge dans `/workspace/data/raw` :
```bash
mkdir -p /workspace/data/raw && cd /workspace/data/raw
# colle ici les commandes wget/curl fournies par AIMI
```

**Option B — miroir Kaggle (plus rapide à scripter)**
```bash
pip install kaggle
# place ton kaggle.json dans ~/.kaggle/ (chmod 600)
kaggle datasets download -d kmader/rsna-bone-age -p /workspace/data/raw
cd /workspace/data/raw && unzip -q rsna-bone-age.zip
```

Arborescence attendue après extraction :
```
data/raw/
├── boneage-training-dataset/        # ~12 611 PNG
├── boneage-training-dataset.csv     # id, boneage(mois), male(bool)
├── boneage-validation-dataset/      # ~1 425 PNG
└── (test set RSNA 200 img si AIMI)
```

> Si les noms diffèrent, ajuste `src/config.py` (chemins) — rien d'autre à changer.

## 3. Pipeline Phase 0/1

```bash
# J1 — exploration
python explore_data.py

# J2 — prétraitement + split (génère data/processed/ + splits.csv)
python preprocess.py

# J3-6 — entraînement baseline
python train.py

# Après l'entraînement — l'EXAMEN : mesure la précision du modèle
python predict.py
```

Sorties : checkpoints dans `runs/`, logs MAE par epoch, meilleur modèle `runs/best.pt`.

---

## Fichiers

- `src/config.py` — chemins, hyperparamètres (point d'entrée à éditer).
- `src/explore_data.py` — EDA : distributions, contrôle qualité.
- `src/preprocess.py` — resize, normalisation, CLAHE optionnel, split train/val.
- `src/dataset.py` — `Dataset` PyTorch (image + sexe → âge en mois).
- `src/model.py` — EfficientNetV2-S + tête de fusion sexe.
- `src/train.py` — boucle d'entraînement régression (Huber), AMP, early stopping.
- `src/predict.py` — l'examen : recharge le modèle, mesure MAE/RMSE/%±1an, graphiques + predictions.csv.

## Notes

- Métrique : **MAE en mois** (comparable au challenge RSNA 2017, gagnant ≈ 4,26 mois).
- Le **sexe est une entrée du modèle** (maturation plus précoce chez les filles).
- Données AIMI/Kaggle : usage recherche/éducation/non-commercial.
