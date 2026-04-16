from __future__ import annotations

import json
import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ..config import AppConfig
from ..data.external import (
    EXTERNAL_DATA_CONTRACT_VERSION,
    ExternalImageDataset,
    ExternalImageRecord,
    load_external_split_records,
)
from ..data.transforms import TransformProfile
from ..model.backbone import TimmBackbone
from ..model.metrics import binary_auroc
from ..utils.logging import log_message
from ..utils.repro import set_global_seed


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


SHARED_WARMUP_DIRNAME = "_shared_external_warmup"
SHARED_WARMUP_REFERENCE_FILENAME = "shared_warmup.json"


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
    set_global_seed(_get_warmup_seed(config))
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
    warmup_metadata = build_external_warmup_metadata(
        config=config,
        backbone_name=backbone_name,
        image_size=image_size,
        transform_profile=transform_profile,
    )
    log_message(
        output_dir,
        "warmup-external: setup"
        f" train_records={len(train_records)}"
        f" val_records={len(val_records)}"
        f" batch_size={config.train.external_warmup_batch_size}"
        f" epochs={config.train.external_warmup_epochs}"
        f" lr={config.train.external_warmup_learning_rate}"
        f" device={config.runtime.device}"
        f" sampler={config.train.external_sampler}"
        f" metadata_hash={warmup_metadata['metadata_hash']}",
    )

    use_cuda = str(config.runtime.device) in ("cuda",) or str(config.runtime.device).startswith("cuda:")
    train_loader = build_external_loader(
        train_records,
        image_size=image_size,
        batch_size=config.train.external_warmup_batch_size,
        num_workers=config.train.external_warmup_num_workers,
        training=True,
        transform_profile=transform_profile,
        sampler_mode=config.train.external_sampler,
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
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        torch.set_float32_matmul_precision("high")
        model.to(memory_format=torch.channels_last)
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
                    "warmup_metadata": warmup_metadata,
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_metadata = build_external_warmup_metadata(
            config=config,
            backbone_name=backbone_name,
            image_size=image_size,
            transform_profile=transform_profile,
        )
        shared_output_dir = _resolve_shared_warmup_output_dir(
            output_dir,
            metadata_hash=str(expected_metadata["metadata_hash"]),
        )
        shared_checkpoint_path = shared_output_dir / "checkpoints" / "best.pt"
        if _checkpoint_matches_metadata(shared_checkpoint_path, expected_metadata):
            _record_shared_warmup_reference(
                output_dir=output_dir,
                shared_output_dir=shared_output_dir,
                checkpoint_path=shared_checkpoint_path,
                metadata=expected_metadata,
            )
            log_message(
                output_dir,
                f"warmup-external: reusing shared checkpoint={shared_checkpoint_path}",
            )
            return shared_checkpoint_path
        if _checkpoint_matches_metadata(checkpoint_path, expected_metadata):
            _promote_warmup_artifacts(source_dir=output_dir, target_dir=shared_output_dir)
            _record_shared_warmup_reference(
                output_dir=output_dir,
                shared_output_dir=shared_output_dir,
                checkpoint_path=shared_checkpoint_path,
                metadata=expected_metadata,
            )
            log_message(
                output_dir,
                f"warmup-external: promoted local checkpoint to shared cache={shared_checkpoint_path}",
            )
            return shared_checkpoint_path
        existing_dir = _find_matching_existing_warmup_dir(
            output_dir=output_dir,
            metadata=expected_metadata,
        )
        if existing_dir is not None:
            _promote_warmup_artifacts(source_dir=existing_dir, target_dir=shared_output_dir)
            _record_shared_warmup_reference(
                output_dir=output_dir,
                shared_output_dir=shared_output_dir,
                checkpoint_path=shared_checkpoint_path,
                metadata=expected_metadata,
            )
            log_message(
                output_dir,
                f"warmup-external: reusing existing checkpoint from={existing_dir}",
            )
            return shared_checkpoint_path
        log_message(
            output_dir,
            "warmup-external: checkpoint metadata mismatch, rebuilding warmup checkpoint",
        )
    else:
        expected_metadata = build_external_warmup_metadata(
            config=config,
            backbone_name=backbone_name,
            image_size=image_size,
            transform_profile=transform_profile,
        )
        shared_output_dir = _resolve_shared_warmup_output_dir(
            output_dir,
            metadata_hash=str(expected_metadata["metadata_hash"]),
        )
        shared_checkpoint_path = shared_output_dir / "checkpoints" / "best.pt"
        existing_dir = _find_matching_existing_warmup_dir(
            output_dir=output_dir,
            metadata=expected_metadata,
        )
        if existing_dir is not None:
            _promote_warmup_artifacts(source_dir=existing_dir, target_dir=shared_output_dir)
            _record_shared_warmup_reference(
                output_dir=output_dir,
                shared_output_dir=shared_output_dir,
                checkpoint_path=shared_checkpoint_path,
                metadata=expected_metadata,
            )
            log_message(
                output_dir,
                f"warmup-external: reusing existing checkpoint from={existing_dir}",
            )
            return shared_checkpoint_path
    artifacts = run_external_warmup(
        config,
        backbone_name=backbone_name,
        output_dir=shared_output_dir,
        image_size=image_size,
        transform_profile=transform_profile,
    )
    _record_shared_warmup_reference(
        output_dir=output_dir,
        shared_output_dir=shared_output_dir,
        checkpoint_path=artifacts.checkpoint_path,
        metadata=expected_metadata,
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
    sampler_mode: str = "none",
    use_cuda: bool,
) -> DataLoader[ExternalBatch]:
    dataset = ExternalImageDataset(
        records,
        image_size=image_size,
        training=training,
        transform_profile=transform_profile,
    )
    sampler = None
    shuffle = training
    if training and sampler_mode == "dataset_label_balanced":
        sampler = _build_balanced_sampler(records)
        shuffle = False
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_collate_external_samples,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
        progress = tqdm(
            loader,
            desc="warmup-eval",
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )
        for batch in progress:
            labels = batch["label"].to(device, non_blocking=use_cuda)
            logits = model(
                batch["image"].to(
                    device,
                    non_blocking=use_cuda,
                    memory_format=torch.channels_last if use_cuda else torch.contiguous_format,
                )
            )
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits.float()).detach().cpu().tolist()
            total_loss += float(loss.float().detach()) * int(labels.numel())
            total_items += int(labels.numel())
            targets.extend(int(value) for value in labels.detach().cpu().tolist())
            scores.extend(float(value) for value in probs)
            progress.set_postfix(loss=f"{float(loss.float().detach()):.4f}")
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
    progress = tqdm(
        loader,
        desc="warmup-train",
        leave=False,
        dynamic_ncols=True,
        disable=False,
    )
    for batch in progress:
        labels = batch["label"].to(device, non_blocking=use_cuda)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            logits = model(
                batch["image"].to(
                    device,
                    non_blocking=use_cuda,
                    memory_format=torch.channels_last if use_cuda else torch.contiguous_format,
                )
            )
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()  # type: ignore[arg-type]
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.float().detach()) * int(labels.numel())
        total_items += int(labels.numel())
        progress.set_postfix(loss=f"{float(loss.float().detach()):.4f}")
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


def _build_balanced_sampler(
    records: Sequence[ExternalImageRecord],
) -> WeightedRandomSampler:
    counts: dict[tuple[str, int], int] = {}
    for record in records:
        key = (record.dataset, record.label)
        counts[key] = counts.get(key, 0) + 1
    weights = [1.0 / counts[(record.dataset, record.label)] for record in records]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(records),
        replacement=True,
    )


def build_external_warmup_metadata(
    *,
    config: AppConfig,
    backbone_name: str,
    image_size: int,
    transform_profile: TransformProfile,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "seed": _get_warmup_seed(config),
        "backbone_name": backbone_name,
        "image_size": image_size,
        "transform_profile": transform_profile,
        "data_contract_version": EXTERNAL_DATA_CONTRACT_VERSION,
        "data_signature": _build_external_data_signature(config),
        "external_warmup_epochs": config.train.external_warmup_epochs,
        "external_warmup_batch_size": config.train.external_warmup_batch_size,
        "external_warmup_num_workers": config.train.external_warmup_num_workers,
        "external_warmup_learning_rate": config.train.external_warmup_learning_rate,
        "external_warmup_max_samples": config.train.external_warmup_max_samples,
        "external_sampler": config.train.external_sampler,
    }
    payload["metadata_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _get_warmup_seed(config: AppConfig) -> int:
    return getattr(config.runtime, "warmup_seed", None) or config.runtime.seed


def _build_external_data_signature(config: AppConfig) -> str:
    path_items: list[dict[str, object]] = []
    paths_obj = getattr(config, "paths", None)
    catalog = getattr(paths_obj, "external_catalog", None)
    splits_dir = getattr(paths_obj, "external_splits_dir", None)
    for raw_path in (
        catalog,
        Path(splits_dir) / "train.csv" if splits_dir is not None else None,
        Path(splits_dir) / "val.csv" if splits_dir is not None else None,
        Path(splits_dir) / "test.csv" if splits_dir is not None else None,
    ):
        if raw_path is None:
            continue
        path = Path(raw_path)
        item: dict[str, object] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.is_file():
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            item["size"] = path.stat().st_size
        path_items.append(item)
    return hashlib.sha256(
        json.dumps(
            {
                "contract_version": EXTERNAL_DATA_CONTRACT_VERSION,
                "files": path_items,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_shared_warmup_output_dir(output_dir: Path, *, metadata_hash: str) -> Path:
    return output_dir.parent.parent / SHARED_WARMUP_DIRNAME / metadata_hash


def _checkpoint_matches_metadata(
    checkpoint_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_metadata = checkpoint.get("warmup_metadata")
    return isinstance(saved_metadata, dict) and saved_metadata == expected_metadata


def _record_shared_warmup_reference(
    *,
    output_dir: Path,
    shared_output_dir: Path,
    checkpoint_path: Path,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / SHARED_WARMUP_REFERENCE_FILENAME
    reference_payload = {
        "shared_output_dir": str(shared_output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metadata_hash": metadata["metadata_hash"],
    }
    reference_path.write_text(json.dumps(reference_payload, indent=2), encoding="utf-8")
    shared_metrics = shared_output_dir / "metrics.json"
    if shared_metrics.exists():
        shutil.copy2(shared_metrics, output_dir / "metrics.json")


def _promote_warmup_artifacts(*, source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in (
        Path("checkpoints") / "best.pt",
        Path("metrics.json"),
        Path("run.log"),
    ):
        source_path = source_dir / relative_path
        if not source_path.exists():
            continue
        destination_path = target_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _find_matching_existing_warmup_dir(
    *,
    output_dir: Path,
    metadata: dict[str, object],
) -> Path | None:
    runs_root = output_dir.parent.parent
    shared_root = runs_root / SHARED_WARMUP_DIRNAME
    candidate_paths = list(runs_root.glob("*/external_warmup/checkpoints/best.pt"))
    candidate_paths.extend(shared_root.glob("*/checkpoints/best.pt"))
    for checkpoint_path in candidate_paths:
        candidate_dir = checkpoint_path.parent.parent
        if candidate_dir == output_dir:
            continue
        if _checkpoint_matches_metadata(checkpoint_path, metadata):
            if candidate_dir.resolve() == _resolve_shared_warmup_output_dir(
                output_dir,
                metadata_hash=str(metadata["metadata_hash"]),
            ).resolve():
                return candidate_dir
            return candidate_dir
    return None
