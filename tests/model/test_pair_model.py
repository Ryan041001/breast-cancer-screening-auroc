from __future__ import annotations

import torch

from final_project.model.fusion import PairedBreastModel
from final_project.model.losses import build_binary_loss
from final_project.model.metrics import binary_auroc


def test_pair_model_returns_one_logit_per_breast() -> None:
    model = PairedBreastModel(backbone_name="resnet18", pretrained=False)
    cc_images = torch.randn(2, 3, 64, 64)
    mlo_images = torch.randn(2, 3, 64, 64)

    logits = model(cc_images, mlo_images)

    assert tuple(logits.shape) == (2,)


def test_auc_metric_accepts_breast_level_predictions() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.7, 0.9]

    auc = binary_auroc(y_true, y_score)

    assert auc == 1.0


def test_binary_loss_supports_optional_positive_class_weight() -> None:
    criterion = build_binary_loss(pos_weight=3.0)
    logits = torch.tensor([0.0, 0.5], dtype=torch.float32)
    targets = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert float(loss) > 0.0
