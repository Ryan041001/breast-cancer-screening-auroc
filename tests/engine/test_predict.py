from __future__ import annotations

from pathlib import Path

import torch

from final_project.engine.predict import load_model_from_checkpoint
from final_project.engine.trainer import Trainer
from final_project.model.fusion import PairedBreastModel


def test_load_model_from_checkpoint_uses_saved_backbone_name(tmp_path: Path) -> None:
    model = PairedBreastModel(backbone_name="resnet18", pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, output_dir=tmp_path)

    checkpoint_path = trainer.save_best_checkpoint(epoch=1, metric=0.5)

    loaded_model = load_model_from_checkpoint(
        checkpoint_path,
        device="cpu",
        backbone_name="efficientnet_b0",
    )

    assert isinstance(loaded_model, PairedBreastModel)
    assert loaded_model.backbone_name == "resnet18"
