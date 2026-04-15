from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .backbone import TimmBackbone


@dataclass(frozen=True, slots=True)
class FusionHeadConfig:
    """Serializable config for the fusion head variant."""

    variant: str = "linear"
    hidden_dim: int = 512
    dropout: float = 0.0
    activation: str = "gelu"
    layer_norm: bool = False
    residual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FusionHeadConfig:
        return cls(
            variant=str(data.get("variant", "linear")),
            hidden_dim=int(data.get("hidden_dim", 512)),
            dropout=float(data.get("dropout", 0.0)),
            activation=str(data.get("activation", "gelu")),
            layer_norm=bool(data.get("layer_norm", False)),
            residual=bool(data.get("residual", False)),
        )

    @classmethod
    def from_train_config(cls, train_config: object) -> FusionHeadConfig:
        """Build from a TrainConfig dataclass (avoids circular import)."""
        return cls(
            variant=getattr(train_config, "fusion_head_variant", "linear"),
            hidden_dim=getattr(train_config, "fusion_hidden_dim", 512),
            dropout=getattr(train_config, "fusion_dropout", 0.0),
            activation=getattr(train_config, "fusion_activation", "gelu"),
            layer_norm=getattr(train_config, "fusion_layer_norm", False),
            residual=getattr(train_config, "fusion_residual", False),
        )

    @classmethod
    def baseline(cls) -> FusionHeadConfig:
        """Return the baseline linear head config."""
        return cls()


def _get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


class LinearFusionHead(nn.Module):
    """Baseline: single Linear(4*D, 1)."""

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


class MLPFusionHead(nn.Module):
    """MLP fusion head with optional LayerNorm, Dropout, and residual."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        input_dim = feature_dim * 4
        layers: list[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.residual = residual
        if residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

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
        hidden = self.trunk(fused)
        if self.residual:
            hidden = hidden + self.residual_proj(fused)
        return self.output(hidden).squeeze(1)


def build_fusion_head(
    feature_dim: int,
    config: FusionHeadConfig | None = None,
) -> nn.Module:
    """Factory: build the fusion head from config."""
    if config is None:
        config = FusionHeadConfig.baseline()
    if config.variant == "mlp":
        return MLPFusionHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm=config.layer_norm,
            residual=config.residual,
        )
    return LinearFusionHead(feature_dim=feature_dim)


# Keep backward compat alias
FusionHead = LinearFusionHead


class PairedBreastModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_chans: int = 3,
        fusion_head_config: FusionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = TimmBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        if fusion_head_config is None:
            fusion_head_config = FusionHeadConfig.baseline()
        self.fusion_head_config = fusion_head_config
        self.head = build_fusion_head(
            feature_dim=self.backbone.num_features,
            config=fusion_head_config,
        )

    def forward(
        self, cc_images: torch.Tensor, mlo_images: torch.Tensor
    ) -> torch.Tensor:
        cc_features = self.backbone(cc_images)
        mlo_features = self.backbone(mlo_images)
        return self.head(cc_features, mlo_features)
