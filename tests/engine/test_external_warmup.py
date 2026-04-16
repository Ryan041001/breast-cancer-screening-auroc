from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import WeightedRandomSampler

from final_project.engine.external_warmup import (
    ExternalWarmupArtifacts,
    build_external_loader,
    build_external_warmup_metadata,
    maybe_prepare_external_warmup,
)


def _make_config(**train_overrides: object) -> SimpleNamespace:
    defaults = dict(
        external_warmup_epochs=4,
        external_warmup_batch_size=32,
        external_warmup_num_workers=4,
        external_warmup_learning_rate=5e-4,
        external_warmup_max_samples=None,
        external_sampler="dataset_label_balanced",
    )
    defaults.update(train_overrides)
    return SimpleNamespace(
        runtime=SimpleNamespace(seed=7),
        train=SimpleNamespace(**defaults),
    )


def test_build_external_loader_uses_balanced_sampler_for_training(
    monkeypatch,
) -> None:
    class FakeDataset:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "sample_id": f"S{index}",
                "image": torch.zeros(3, 8, 8),
                "label": torch.tensor(float(index % 2)),
            }

    monkeypatch.setattr(
        "final_project.engine.external_warmup.ExternalImageDataset",
        FakeDataset,
    )

    records = [
        SimpleNamespace(dataset="vindr", label=0),
        SimpleNamespace(dataset="vindr", label=0),
        SimpleNamespace(dataset="cmmd", label=1),
        SimpleNamespace(dataset="cmmd", label=1),
    ]

    loader = build_external_loader(
        records,
        image_size=32,
        batch_size=2,
        num_workers=0,
        training=True,
        transform_profile="baseline",
        sampler_mode="dataset_label_balanced",
        use_cuda=False,
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_maybe_prepare_external_warmup_reuses_checkpoint_when_metadata_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _make_config()
    output_dir = tmp_path / "runs" / "demo" / "external_warmup"
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    metadata = build_external_warmup_metadata(
        config=config,
        backbone_name="efficientnet_b0",
        image_size=384,
        transform_profile="baseline",
    )
    torch.save({"warmup_metadata": metadata}, checkpoint_path)

    def fail_run(*args, **kwargs):
        raise AssertionError("run_external_warmup should not be called")

    monkeypatch.setattr(
        "final_project.engine.external_warmup.run_external_warmup",
        fail_run,
    )

    reused = maybe_prepare_external_warmup(
        config,
        backbone_name="efficientnet_b0",
        output_dir=output_dir,
        image_size=384,
        transform_profile="baseline",
    )

    assert reused is not None
    assert reused.name == "best.pt"
    assert reused != checkpoint_path
    assert reused.exists()
    reference = json.loads(
        (output_dir / "shared_warmup.json").read_text(encoding="utf-8")
    )
    assert reference["checkpoint_path"] == str(reused.resolve())


def test_maybe_prepare_external_warmup_rebuilds_checkpoint_when_metadata_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _make_config()
    output_dir = tmp_path / "runs" / "demo" / "external_warmup"
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    stale_config = _make_config(external_warmup_learning_rate=1e-3)
    stale_metadata = build_external_warmup_metadata(
        config=stale_config,
        backbone_name="efficientnet_b0",
        image_size=384,
        transform_profile="baseline",
    )
    torch.save({"warmup_metadata": stale_metadata}, checkpoint_path)

    rebuilt_path = output_dir / "checkpoints" / "rebuilt.pt"
    calls: list[Path] = []

    def fake_run(*args, **kwargs) -> ExternalWarmupArtifacts:
        calls.append(kwargs["output_dir"])
        return ExternalWarmupArtifacts(
            checkpoint_path=rebuilt_path,
            metrics_path=output_dir / "metrics.json",
        )

    monkeypatch.setattr(
        "final_project.engine.external_warmup.run_external_warmup",
        fake_run,
    )

    resolved = maybe_prepare_external_warmup(
        config,
        backbone_name="efficientnet_b0",
        output_dir=output_dir,
        image_size=384,
        transform_profile="baseline",
    )

    expected_shared = (
        output_dir.parent.parent / "_shared_external_warmup" / build_external_warmup_metadata(
            config=config,
            backbone_name="efficientnet_b0",
            image_size=384,
            transform_profile="baseline",
        )["metadata_hash"]
    )
    assert calls == [expected_shared]
    assert resolved == rebuilt_path


def test_maybe_prepare_external_warmup_reuses_matching_checkpoint_from_other_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _make_config()
    runs_root = tmp_path / "runs"
    source_output_dir = runs_root / "source_exp" / "external_warmup"
    source_checkpoint = source_output_dir / "checkpoints" / "best.pt"
    source_checkpoint.parent.mkdir(parents=True)
    metadata = build_external_warmup_metadata(
        config=config,
        backbone_name="efficientnet_b0",
        image_size=384,
        transform_profile="baseline",
    )
    torch.save({"warmup_metadata": metadata}, source_checkpoint)
    (source_output_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.9}),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("run_external_warmup should not be called")

    monkeypatch.setattr(
        "final_project.engine.external_warmup.run_external_warmup",
        fail_run,
    )

    target_output_dir = runs_root / "target_exp" / "external_warmup"
    reused = maybe_prepare_external_warmup(
        config,
        backbone_name="efficientnet_b0",
        output_dir=target_output_dir,
        image_size=384,
        transform_profile="baseline",
    )

    assert reused is not None
    assert reused.exists()
    assert str(reused).endswith("checkpoints\\best.pt")
    assert json.loads(
        (target_output_dir / "shared_warmup.json").read_text(encoding="utf-8")
    )["checkpoint_path"] == str(reused.resolve())
