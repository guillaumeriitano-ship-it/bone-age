"""Configuration centrale — édite uniquement ce fichier pour adapter chemins/hyperparamètres."""
from pathlib import Path

# --- Chemins -------------------------------------------------------------
# Racine projet (par défaut /workspace/bone-age sur RunPod)
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RUNS = ROOT / "runs"

# Dossiers/fichiers RSNA (ajuste si les noms diffèrent après extraction)
TRAIN_IMG_DIR = DATA_RAW / "boneage-training-dataset"
TRAIN_CSV = DATA_RAW / "boneage-training-dataset.csv"
VAL_IMG_DIR = DATA_RAW / "boneage-validation-dataset"
# Le CSV de validation Kaggle s'appelle parfois "Validation Dataset.csv"
VAL_CSV = DATA_RAW / "boneage-validation-dataset.csv"

# Noms de colonnes dans le CSV d'entraînement
COL_ID = "id"
COL_AGE = "boneage"   # en mois
COL_MALE = "male"     # booléen / True-False

# --- Prétraitement -------------------------------------------------------
IMG_SIZE = 512
USE_CLAHE = True      # égalisation adaptative pour rehausser les contours osseux
VAL_FRACTION = 0.15   # part du train utilisée en validation interne si pas de val labellisée
SEED = 42

# --- Entraînement --------------------------------------------------------
MODEL_NAME = "tf_efficientnetv2_m.in21k_ft_in1k"
BATCH_SIZE = 24
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
HUBER_DELTA = 10.0    # robustesse aux outliers (en mois)
EARLY_STOP_PATIENCE = 8
AMP = True            # mixed precision

# Normalisation de la cible : on standardise l'âge (mois) pour stabiliser la régression
# (les stats réelles sont recalculées sur le train dans train.py)
AGE_MEAN = 127.0      # ~moyenne RSNA (mois)
AGE_STD = 41.0        # ~écart-type RSNA (mois)
