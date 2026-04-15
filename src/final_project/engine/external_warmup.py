from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..config import AppConfig
from ..data.external import (
    ExternalImageDataset,
    ExternalImageRecord,
    load_external_split_records,
)
from ..data.transforms import TransformProfile
from ..model.backbone import TimmBackbone
from ..model.metrics import binary_auroc
from ..utils.logging import log_message


class ExternalBatch(TypedDict):
    sample_id: list[str]
    image: torch.Tensor
    label: torch.Tensor


@dataclass(frozen=True, slots=True)
class ExternalWarmupArtifacts:
    checkpoint_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class ExternalWarmupResult:
    loss: float
    auc: float


class ExternalWarmupModel(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = TimmBackbone(backbone_name=backbone_name, pretrained=pretrained)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)).squeeze(1)


def run_external_warmup(
    config: AppConfig,
    *,
    backbone_name: str,
    output_dir: Path,
    image_size: int,
    transform_profile: TransformProfile,
) -> ExternalWarmupArtifacts:
    if config.paths.external_data_root is None:
        raise ValueError("External warmup requires 'paths.external_data_root'")
    if config.paths.external_catalog is None or config.paths.external_splits_dir is None:
        raise ValueError(
            "External warmup requires resolved external catalog and splits paths"
        )

    train_records = load_external_split_records(
        catalog_csv=config.paths.external_catalog,
        processed_root=config.paths.external_data_root / "processed",
        split_csv=config.paths.external_splits_dir / "train.csv",
        max_samples=config.train.external_warmup_max_samples,
    )
    val_records = load_external_split_records(
        catalog_csv=config.paths.external_catalog,
        processed_root=config.paths.external_data_root / "processed",
        split_csv=config.paths.external_splits_dir / "val.csv",
        max_samples=max(
            1,
            config.train.external_warmup_max_samples // 4,
        )
        if config.train.external_warmup_max_samples is not None
        else None,
    )
    if not train_records or not val_records:
        raise ValueError("External warmup requires non-empty train/val splits")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    log_message(
        output_dir,
        "warmup-external: setup"
        f" train_records={len(train_records)}"
        f" val_records={len(val_records)}"
        f" batch_size={config.train.external_warmup_batch_size}"
        f" epochs={config.train.external_warmup_epochs}"
        f" lr={config.train.external_warmup_learning_rate}"
        f" device={config.runtime.device}",
    )

    use_cuda = str(config.runtime.device) in ("cuda",) or str(config.runtime.device).startswith("cuda:")
    train_loader = build_external_loader(
        train_records,
        image_size=image_size,
        batch_size=config.train.external_warmup_batch_size,
        num_workers=config.train.external_warmup_num_workers,
        training=True,
        transform_profile=transform_profile,
        use_cuda=use_cuda,
    )
    val_loader = build_external_loader(
        val_records,
        image_size=image_size,
        batch_size=config.train.external_warmup_batch_size,
        num_workers=config.train.external_warmup_num_workers,
        training=False,
        transform_profile=transform_profile,
        use_cuda=use_cuda,
    )
    model = ExternalWarmupModel(backbone_name=backbone_name, pretrained=True)
    model.to(config.runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.external_warmup_learning_rate,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=_compute_positive_class_weight(train_records)
    ).to(config.runtime.device)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    best_auc = float("-inf")
    best_metrics = {"loss": 0.0, "auc": 0.0, "epochs": config.train.external_warmup_epochs}
    for epoch in range(config.train.external_warmup_epochs):
        log_message(
            output_dir,
            f"warmup-external: epoch {epoch + 1}/{config.train.external_warmup_epochs} start",
        )
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.runtime.device,
            use_cuda=use_cuda,
            scaler=scaler,
        )
        evaluation = evaluate_external_model(
            model,
            val_loader,
            criterion,
            config.runtime.device,
            use_cuda=use_cuda,
        )
        log_message(
            output_dir,
            f"warmup-external: epoch {epoch + 1}/{config.train.external_warmup_epochs} done"
            f" train_loss={train_loss:.6f}"
            f" val_loss={evaluation.loss:.6f}"
            f" val_auc={evaluation.auc:.6f}",
        )
        if evaluation.auc >= best_auc:
            best_auc = evaluation.auc
            best_metrics = {
                "loss": evaluation.loss,
                "auc": evaluation.auc,
                "epochs": config.train.external_warmup_epochs,
            }
            torch.save(
                {
                    "backbone_name": backbone_name,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "metric": evaluation.auc,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
            log_message(
                output_dir,
                f"warmup-external: checkpoint updated path={checkpoint_path} metric={evaluation.auc:.6f}",
            )
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    log_message(
        output_dir,
        f"warmup-external: complete checkpoint={checkpoint_path} best_auc={best_metrics['auc']:.6f}",
    )
    return ExternalWarmupArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


def maybe_prepare_external_warmup(
    config: AppConfig,
    *,
    backbone_name: str,
    output_dir: Path,
    image_size: int,
    transform_profile: TransformProfile,
) -> Path | None:
    if config.train.external_warmup_epochs <= 0:
        return None
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    if checkpoint_path.exists():
        log_message(output_dir, f"warmup-external: reusing checkpoint={checkpoint_path}")
        return checkpoint_path
    artifacts = run_external_warmup(
        config,
        backbone_name=backbone_name,
        output_dir=output_dir,
        image_size=image_size,
        transform_profile=transform_profile,
    )
    return artifacts.checkpoint_path


def load_backbone_from_warmup(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    backbone_state = checkpoint.get("backbone_state_dict")
    if not isinstance(backbone_state, dict):
        raise ValueError("Warmup checkpoint is missing 'backbone_state_dict'")
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Target model does not expose a 'backbone' attribute")
    backbone.load_state_dict(backbone_state)


def build_external_loader(
    records: Sequence[ExternalImageRecord],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    training: bool,
    transform_profile: TransformProfile,
    use_cuda: bool,
) -> DataLoader[ExternalBatch]:
    dataset = ExternalImageDataset(
        records,
        image_size=image_size,
        training=training,
        transform_profile=transform_profile,
    )
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            collate_fn=_collate_external_samples,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        collate_fn=_collate_external_samples,
        pin_memory=use_cuda,
    )


def evaluate_external_model(
    model: nn.Module,
    loader: DataLoader[ExternalBatch],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    use_cuda: bool = False,
) -> ExternalWarmupResult:
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets: list[int] = []
    scores: list[float] = []
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda),
    ):
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=use_cuda)
            logits = model(batch["image"].to(device, non_blocking=use_cuda))
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits.float()).detach().cpu().tolist()
            total_loss += float(loss.float().detach()) * int(labels.numel())
            total_items += int(labels.numel())
            targets.extend(int(value) for value in labels.detach().cpu().tolist())
            scores.extend(float(value) for value in probs)
    average_loss = total_loss / total_items if total_items else 0.0
    auc = binary_auroc(targets, scores) if len(set(targets)) > 1 else 0.5
    return ExternalWarmupResult(loss=average_loss, auc=auc)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[ExternalBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    use_cuda: bool,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=use_cuda)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            logits = model(batch["image"].to(device, non_blocking=use_cuda))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()  # type: ignore[arg-type]
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.float().detach()) * int(labels.numel())
        total_items += int(labels.numel())
    return total_loss / total_items if total_items else 0.0


def _collate_external_samples(samples: Sequence[dict[str, torch.Tensor | str]]) -> ExternalBatch:
    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "image": torch.stack([sample["image"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]).float(),
    }


def _compute_positive_class_weight(records: Sequence[ExternalImageRecord]) -> torch.Tensor | None:
    positives = sum(1 for record in records if record.label == 1)
    negatives = sum(1 for record in records if record.label == 0)
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor(negatives / positives, dtype=torch.float32)
