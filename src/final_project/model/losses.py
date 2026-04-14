from __future__ import annotations

import torch
from torch import nn


def build_binary_loss(pos_weight: float | None = None) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]))
