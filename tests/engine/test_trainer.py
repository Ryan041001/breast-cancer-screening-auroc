from __future__ import annotations

from pathlib import Path

import torch
import pytest
from PIL import Image
from torch import nn

from final_project.data.manifest import BreastManifestRecord
from final_project.engine.trainer import Trainer
from final_project.engine.trainer import fit_model
from final_project.model.fusion import PairedBreastModel


def test_trainer_saves_and_reloads_best_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, output_dir=tmp_path)

    with torch.no_grad():
        model.weight.fill_(1.5)
        model.bias.fill_(0.25)

    checkpoint_path = trainer.save_best_checkpoint(epoch=3, metric=0.81)
    saved_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    restored = trainer.load_checkpoint(checkpoint_path)

    assert checkpoint_path.name == "best.pt"
    assert restored["epoch"] == 3
    assert restored["metric"] == 0.81
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved_state[name])


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


class TinyPairedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, cc_image: torch.Tensor, mlo_image: torch.Tensor) -> torch.Tensor:
        return cc_image.mean(dim=(1, 2, 3)) + mlo_image.mean(dim=(1, 2, 3)) + self.bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fit_model_runs_on_cuda(tmp_path: Path) -> None:
    train_records: list[BreastManifestRecord] = []
    val_records: list[BreastManifestRecord] = []
    for breast_id, label, bucket, pixel in [
        ("100_L", 0, train_records, 16),
        ("101_R", 1, train_records, 220),
        ("102_L", 0, val_records, 24),
        ("103_R", 1, val_records, 232),
    ]:
        cc_path = tmp_path / f"{breast_id}_CC.jpg"
        mlo_path = tmp_path / f"{breast_id}_MLO.jpg"
        _write_image(cc_path, pixel)
        _write_image(mlo_path, pixel)
        bucket.append(
            BreastManifestRecord(
                breast_id=breast_id,
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=label,
            )
        )

    model = PairedBreastModel(backbone_name="resnet18", pretrained=False)

    trainer, evaluation = fit_model(
        model,
        train_records,
        val_records,
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        device="cuda",
        output_dir=tmp_path / "cuda_fit",
    )

    assert (trainer.checkpoints_dir / "best.pt").exists()
    assert len(evaluation.predictions) == 2


def test_fit_model_reports_epoch_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_records: list[BreastManifestRecord] = []
    val_records: list[BreastManifestRecord] = []
    for breast_id, label, bucket, pixel in [
        ("100_L", 0, train_records, 16),
        ("101_R", 1, train_records, 220),
        ("102_L", 0, val_records, 24),
        ("103_R", 1, val_records, 232),
    ]:
        cc_path = tmp_path / f"{breast_id}_CC.jpg"
        mlo_path = tmp_path / f"{breast_id}_MLO.jpg"
        _write_image(cc_path, pixel)
        _write_image(mlo_path, pixel)
        bucket.append(
            BreastManifestRecord(
                breast_id=breast_id,
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=label,
            )
        )

    trainer, evaluation = fit_model(
        TinyPairedModel(),
        train_records,
        val_records,
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=2,
        device="cpu",
        output_dir=tmp_path / "cpu_fit",
    )

    command_output = capsys.readouterr().out
    assert "train: epoch 1/2 start" in command_output
    assert "train: epoch 1/2 done" in command_output
    assert "train: epoch 2/2 done" in command_output
    assert "val_auc=" in command_output
    assert (trainer.checkpoints_dir / "best.pt").exists()
    assert len(evaluation.predictions) == 2
