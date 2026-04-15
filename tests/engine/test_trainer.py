from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import pytest
from PIL import Image
from torch import nn

from final_project.data.manifest import BreastManifestRecord
from final_project.engine.trainer import EvaluationResult, Trainer
from final_project.engine.trainer import (
    build_training_loader,
    fit_full_model,
    fit_model,
)
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


def _make_record(
    tmp_path: Path, breast_id: str, label: int | None
) -> BreastManifestRecord:
    return BreastManifestRecord(
        breast_id=breast_id,
        cc_path=tmp_path / f"{breast_id}_CC.jpg",
        mlo_path=tmp_path / f"{breast_id}_MLO.jpg",
        label=label,
    )


class TinyPairedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, cc_image: torch.Tensor, mlo_image: torch.Tensor) -> torch.Tensor:
        return cc_image.mean(dim=(1, 2, 3)) + mlo_image.mean(dim=(1, 2, 3)) + self.bias


@pytest.mark.parametrize(
    ("transform_profile", "expected_profile"),
    [(None, "baseline"), ("normaug", "normaug")],
)
def test_build_training_loader_forwards_transform_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transform_profile: Literal["normaug"] | None,
    expected_profile: str,
) -> None:
    observed: dict[str, object] = {}

    class CapturingDataset:
        def __init__(
            self,
            *,
            records,
            image_size,
            training,
            transform_profile,
            cache_mode,
        ) -> None:
            observed.update(
                {
                    "records": list(records),
                    "image_size": image_size,
                    "training": training,
                    "transform_profile": transform_profile,
                    "cache_mode": cache_mode,
                }
            )

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            return {
                "breast_id": "100_L",
                "cc_image": torch.zeros(1, 1, 1),
                "mlo_image": torch.zeros(1, 1, 1),
                "label": torch.tensor(0.0),
            }

    monkeypatch.setattr(
        "final_project.engine.trainer.PairedBreastDataset", CapturingDataset
    )

    records = [_make_record(tmp_path, "100_L", 0)]
    if transform_profile is None:
        build_training_loader(
            records,
            image_size=32,
            batch_size=2,
            num_workers=0,
            training=True,
        )
    else:
        build_training_loader(
            records,
            image_size=32,
            batch_size=2,
            num_workers=0,
            training=True,
            transform_profile=transform_profile,
        )

    assert observed["records"] == records
    assert observed["image_size"] == 32
    assert observed["training"] is True
    assert observed["transform_profile"] == expected_profile
    assert observed["cache_mode"] == "preprocess"


def test_fit_model_passes_transform_profile_to_train_and_val_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_calls: list[tuple[bool, str, bool]] = []

    def fake_build_training_loader(
        records,
        image_size,
        batch_size,
        num_workers,
        training,
        *,
        transform_profile,
        cache_mode,
        use_cuda=False,
    ):
        loader_calls.append((training, transform_profile, cache_mode, use_cuda))
        return object()

    monkeypatch.setattr(
        "final_project.engine.trainer.build_training_loader",
        fake_build_training_loader,
    )
    monkeypatch.setattr(
        "final_project.engine.trainer.build_binary_loss",
        lambda pos_weight: nn.BCEWithLogitsLoss(),
    )
    monkeypatch.setattr(
        "final_project.engine.trainer._compute_positive_class_weight",
        lambda records: None,
    )
    monkeypatch.setattr(
        "final_project.engine.trainer._train_one_epoch",
        lambda model,
        loader,
        optimizer,
        criterion,
        device,
        *,
        use_cuda=False,
        scaler=None,
        grad_accum_steps=1: 0.2,
    )
    monkeypatch.setattr(
        "final_project.engine.trainer.evaluate_model",
        lambda model, loader, criterion, device, *, use_cuda=False: EvaluationResult(
            loss=0.1,
            auc=0.9,
            predictions={"200_L": 0.9},
        ),
    )

    trainer, evaluation = fit_model(
        TinyPairedModel(),
        [_make_record(tmp_path, "100_L", 0), _make_record(tmp_path, "101_R", 1)],
        [_make_record(tmp_path, "200_L", 1)],
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "fit_with_profile",
        transform_profile="normaug",
    )

    assert loader_calls == [
        (True, "normaug", "preprocess", False),
        (False, "normaug", "preprocess", False),
    ]
    assert (trainer.checkpoints_dir / "best.pt").exists()
    assert evaluation.predictions == {"200_L": 0.9}


def test_fit_full_model_passes_transform_profile_to_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_calls: list[tuple[bool, str, bool]] = []

    def fake_build_training_loader(
        records,
        image_size,
        batch_size,
        num_workers,
        training,
        *,
        transform_profile,
        cache_mode,
        use_cuda=False,
    ):
        loader_calls.append((training, transform_profile, cache_mode, use_cuda))
        return object()

    monkeypatch.setattr(
        "final_project.engine.trainer.build_training_loader",
        fake_build_training_loader,
    )
    monkeypatch.setattr(
        "final_project.engine.trainer.build_binary_loss",
        lambda pos_weight: nn.BCEWithLogitsLoss(),
    )
    monkeypatch.setattr(
        "final_project.engine.trainer._compute_positive_class_weight",
        lambda records: None,
    )
    monkeypatch.setattr(
        "final_project.engine.trainer._train_one_epoch",
        lambda model,
        loader,
        optimizer,
        criterion,
        device,
        *,
        use_cuda=False,
        scaler=None,
        grad_accum_steps=1: 0.2,
    )

    trainer = fit_full_model(
        TinyPairedModel(),
        [_make_record(tmp_path, "100_L", 0), _make_record(tmp_path, "101_R", 1)],
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "full_fit_with_profile",
        transform_profile="normaug",
    )

    assert loader_calls == [(True, "normaug", "preprocess", False)]
    assert (trainer.checkpoints_dir / "best.pt").exists()


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
    log_text = (tmp_path / "cpu_fit" / "run.log").read_text(encoding="utf-8")
    assert "train: setup" in log_text
    assert "train: epoch 1/2 start" in log_text
    assert "train: complete" in log_text
    assert len(evaluation.predictions) == 2
