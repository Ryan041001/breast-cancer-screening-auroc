from __future__ import annotations

import timm
import torch
from torch import nn
from typing import cast


class TimmBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = cast(
            nn.Module,
            timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
            ),
        )
        num_features = getattr(self.encoder, "num_features", None)
        if not isinstance(num_features, int):
            raise ValueError(
                f"Backbone '{backbone_name}' must expose an integer num_features"
            )
        self.num_features = num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
