from __future__ import annotations

import torch
from torch import nn

from .backbone import TimmBackbone


class FusionHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(feature_dim * 4, 1)

    def forward(
        self, cc_features: torch.Tensor, mlo_features: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                cc_features,
                mlo_features,
                torch.abs(cc_features - mlo_features),
                cc_features * mlo_features,
            ],
            dim=1,
        )
        return self.output(fused).squeeze(1)


class PairedBreastModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = TimmBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.head = FusionHead(feature_dim=self.backbone.num_features)

    def forward(
        self, cc_images: torch.Tensor, mlo_images: torch.Tensor
    ) -> torch.Tensor:
        cc_features = self.backbone(cc_images)
        mlo_features = self.backbone(mlo_images)
        return self.head(cc_features, mlo_features)
