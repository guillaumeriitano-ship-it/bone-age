"""Modèle baseline : EfficientNetV2-S (timm) + fusion du sexe -> régression âge.

L'embedding image est concaténé à une petite représentation du sexe, puis une
tête MLP prédit l'âge standardisé (1 valeur). Le sexe améliore systématiquement
la précision (maturation osseuse plus précoce chez les filles).
"""
import timm
import torch
import torch.nn as nn

import config as C


class BoneAgeModel(nn.Module):
    def __init__(self, model_name: str = C.MODEL_NAME, pretrained: bool = True):
        super().__init__()
        # backbone sans tête de classification ; num_classes=0 -> sortie = features
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=3
        )
        feat_dim = self.backbone.num_features

        # encodage du sexe
        self.sex_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, img: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        f = self.backbone(img)            # (B, feat_dim)
        s = self.sex_fc(sex)              # (B, 32)
        x = torch.cat([f, s], dim=1)
        return self.head(x)               # (B, 1) âge standardisé


def denormalize_age(pred_std: torch.Tensor) -> torch.Tensor:
    """Repasse de la cible standardisée vers des mois."""
    return pred_std * C.AGE_STD + C.AGE_MEAN
