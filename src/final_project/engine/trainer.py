from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..data.dataset import PairedBreastDataset, PairedBreastSample
from ..data.manifest import BreastManifestRecord
from ..data.transforms import TransformProfile
from ..model.losses import build_binary_loss
from ..model.metrics import binary_auroc
from ..utils.logging import log_message


class TrainingBatch(TypedDict):
    breast_id: list[str]
    cc_image: torch.Tensor
    mlo_image: torch.Tensor
    label: torch.Tensor


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    auc: float
    predictions: dict[str, float]


@dataclass(slots=True)
class Trainer:
    model: nn.Module
    optimizer: Optimizer
    output_dir: Path
    best_metric: float = field(default=float("-inf"))
    checkpoints_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_best_checkpoint(self, epoch: int, metric: float) -> Path:
        checkpoint_path = self.checkpoints_dir / "best.pt"
        if metric >= self.best_metric:
            self.best_metric = metric
            fusion_head_config = getattr(self.model, "fusion_head_config", None)
            fusion_head_config_dict = (
                fusion_head_config.to_dict()
                if hasattr(fusion_head_config, "to_dict")
                else None
            )
            torch.save(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "backbone_name": getattr(self.model, "backbone_name", None),
                    "fusion_head_config": fusion_head_config_dict,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, object]:
        checkpoint = cast(
            dict[str, object],
            torch.load(
                Path(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            ),
        )
        self.model.load_state_dict(
            cast(dict[str, torch.Tensor], checkpoint["model_state_dict"])
        )
        self.optimizer.load_state_dict(
            cast(dict[str, object], checkpoint["optimizer_state_dict"])
        )
        return {
            "epoch": checkpoint["epoch"],
            "metric": checkpoint["metric"],
        }


def build_training_loader(
    records: Sequence[BreastManifestRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    training: bool,
    *,
    transform_profile: TransformProfile = "baseline",
    use_cuda: bool = False,
) -> DataLoader[TrainingBatch]:
    dataset = PairedBreastDataset(
        records=records,
        image_size=image_size,
        training=training,
        transform_profile=transform_profile,
    )
    if num_workers > 0:
        return cast(
            DataLoader[TrainingBatch],
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=training,
                num_workers=num_workers,
                collate_fn=_collate_training_samples,
                pin_memory=use_cuda,
                persistent_workers=True,
                prefetch_factor=2,
            ),
        )
    return cast(
        DataLoader[TrainingBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            collate_fn=_collate_training_samples,
            pin_memory=use_cuda,
        ),
    )


def fit_model(
    model: nn.Module,
    train_records: Sequence[BreastManifestRecord],
    val_records: Sequence[BreastManifestRecord],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    device: torch.device | str,
    output_dir: Path,
    transform_profile: TransformProfile = "baseline",
    learning_rate: float = 1e-3,
) -> tuple[Trainer, EvaluationResult]:
    use_cuda = str(device) in ("cuda",) or str(device).startswith("cuda:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model=model, optimizer=optimizer, output_dir=output_dir)
    log_message(
        output_dir,
        "train: setup"
        f" train_records={len(train_records)}"
        f" val_records={len(val_records)}"
        f" batch_size={batch_size}"
        f" epochs={epochs}"
        f" device={device}"
        f" transform_profile={transform_profile}",
    )
    criterion = build_binary_loss(
        pos_weight=_compute_positive_class_weight(train_records)
    ).to(device)
    train_loader = build_training_loader(
        train_records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        training=True,
        transform_profile=transform_profile,
        use_cuda=use_cuda,
    )
    val_loader = build_training_loader(
        val_records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        training=False,
        transform_profile=transform_profile,
        use_cuda=use_cuda,
    )

    model.to(device)
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    best_eval = EvaluationResult(loss=float("inf"), auc=float("-inf"), predictions={})
    for epoch in range(epochs):
        log_message(output_dir, f"train: epoch {epoch + 1}/{epochs} start")
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_cuda=use_cuda,
            scaler=scaler,
        )
        eval_result = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            use_cuda=use_cuda,
        )
        log_message(
            output_dir,
            f"train: epoch {epoch + 1}/{epochs} done"
            f" train_loss={train_loss:.6f}"
            f" val_loss={eval_result.loss:.6f}"
            f" val_auc={eval_result.auc:.6f}",
        )
        checkpoint_path = trainer.save_best_checkpoint(
            epoch=epoch + 1, metric=eval_result.auc
        )
        if eval_result.auc >= trainer.best_metric:
            log_message(
                output_dir,
                f"train: checkpoint updated path={checkpoint_path} metric={eval_result.auc:.6f}",
            )
        if eval_result.auc >= best_eval.auc:
            best_eval = eval_result

    trainer.load_checkpoint(trainer.checkpoints_dir / "best.pt")
    log_message(
        output_dir,
        f"train: complete best_auc={best_eval.auc:.6f} checkpoint={trainer.checkpoints_dir / 'best.pt'}",
    )
    return trainer, best_eval


def fit_full_model(
    model: nn.Module,
    train_records: Sequence[BreastManifestRecord],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    device: torch.device | str,
    output_dir: Path,
    transform_profile: TransformProfile = "baseline",
    learning_rate: float = 1e-3,
) -> Trainer:
    use_cuda = str(device) in ("cuda",) or str(device).startswith("cuda:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model=model, optimizer=optimizer, output_dir=output_dir)
    log_message(
        output_dir,
        "train: setup"
        f" train_records={len(train_records)}"
        f" batch_size={batch_size}"
        f" epochs={epochs}"
        f" device={device}"
        f" transform_profile={transform_profile}",
    )
    criterion = build_binary_loss(
        pos_weight=_compute_positive_class_weight(train_records)
    ).to(device)
    train_loader = build_training_loader(
        train_records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        training=True,
        transform_profile=transform_profile,
        use_cuda=use_cuda,
    )

    model.to(device)
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    for epoch in range(epochs):
        log_message(output_dir, f"train: epoch {epoch + 1}/{epochs} start")
        loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_cuda=use_cuda,
            scaler=scaler,
        )
        log_message(
            output_dir,
            f"train: epoch {epoch + 1}/{epochs} done train_loss={loss:.6f}",
        )
        checkpoint_path = trainer.save_best_checkpoint(epoch=epoch + 1, metric=-loss)
        if -loss >= trainer.best_metric:
            log_message(
                output_dir,
                f"train: checkpoint updated path={checkpoint_path} metric={-loss:.6f}",
            )

    trainer.load_checkpoint(trainer.checkpoints_dir / "best.pt")
    log_message(
        output_dir,
        f"train: complete checkpoint={trainer.checkpoints_dir / 'best.pt'}",
    )
    return trainer


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[TrainingBatch],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    use_cuda: bool = False,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets: list[int] = []
    scores: list[float] = []
    predictions: dict[str, float] = {}

    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_cuda,
        ),
    ):
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=use_cuda)
            logits = model(
                batch["cc_image"].to(device, non_blocking=use_cuda),
                batch["mlo_image"].to(device, non_blocking=use_cuda),
            )
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits.float()).detach().cpu().tolist()
            total_loss += float(loss.float().detach()) * int(labels.numel())
            total_items += int(labels.numel())
            targets.extend(int(value) for value in labels.detach().cpu().tolist())
            scores.extend(float(value) for value in probs)
            predictions.update(
                dict(zip(batch["breast_id"], scores[-len(probs) :], strict=True))
            )

    average_loss = total_loss / total_items if total_items else 0.0
    auc = binary_auroc(targets, scores) if len(set(targets)) > 1 else 0.5
    return EvaluationResult(loss=average_loss, auc=auc, predictions=predictions)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[TrainingBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    use_cuda: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=False)
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=use_cuda)
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_cuda,
        ):
            logits = model(
                batch["cc_image"].to(device, non_blocking=use_cuda),
                batch["mlo_image"].to(device, non_blocking=use_cuda),
            )
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()  # type: ignore[arg-type]
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.float().detach()) * int(labels.numel())
        total_items += int(labels.numel())
    return total_loss / total_items if total_items else 0.0


def _collate_training_samples(
    samples: Sequence[PairedBreastSample],
) -> TrainingBatch:
    labels: list[torch.Tensor] = []
    for sample in samples:
        label = sample["label"]
        if label is None:
            raise ValueError("Training samples must include labels")
        labels.append(label)
    return {
        "breast_id": [sample["breast_id"] for sample in samples],
        "cc_image": torch.stack([sample["cc_image"] for sample in samples]),
        "mlo_image": torch.stack([sample["mlo_image"] for sample in samples]),
        "label": torch.stack(labels).float(),
    }


def _compute_positive_class_weight(
    records: Sequence[BreastManifestRecord],
) -> float | None:
    positives = sum(1 for record in records if record.label == 1)
    negatives = sum(1 for record in records if record.label == 0)
    if positives == 0 or negatives == 0:
        return None
    return negatives / positives
