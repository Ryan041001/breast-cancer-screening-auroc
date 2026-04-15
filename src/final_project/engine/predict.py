from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import PairedBreastDataset, PairedBreastSample
from ..data.manifest import BreastManifestRecord
from ..data.transforms import TransformProfile
from ..model.fusion import FusionHeadConfig, PairedBreastModel


DEFAULT_BACKBONE_NAME = "efficientnet_b0"


class PredictionBatch(TypedDict):
    breast_id: list[str]
    cc_image: torch.Tensor
    mlo_image: torch.Tensor


def build_prediction_loader(
    records: Sequence[BreastManifestRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    *,
    transform_profile: TransformProfile = "baseline",
    cache_mode: str = "preprocess",
    use_cuda: bool = False,
) -> DataLoader[PredictionBatch]:
    dataset = PairedBreastDataset(
        records=records,
        image_size=image_size,
        training=False,
        transform_profile=transform_profile,
        cache_mode=cache_mode,
    )
    if num_workers > 0:
        return cast(
            DataLoader[PredictionBatch],
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_prediction_samples,
                pin_memory=use_cuda,
                persistent_workers=True,
                prefetch_factor=2,
            ),
        )
    return cast(
        DataLoader[PredictionBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_prediction_samples,
            pin_memory=use_cuda,
        ),
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    backbone_name: str = DEFAULT_BACKBONE_NAME,
    fusion_head_config: FusionHeadConfig | None = None,
) -> nn.Module:
    checkpoint = cast(
        dict[str, object],
        torch.load(Path(checkpoint_path), map_location=device, weights_only=True),
    )
    resolved_backbone_name = checkpoint.get("backbone_name")
    if not isinstance(resolved_backbone_name, str):
        resolved_backbone_name = backbone_name

    # Prefer checkpoint-saved config, fallback to caller-provided, then baseline
    saved_config_dict = checkpoint.get("fusion_head_config")
    if isinstance(saved_config_dict, dict):
        resolved_fusion_config = FusionHeadConfig.from_dict(saved_config_dict)
    elif fusion_head_config is not None:
        resolved_fusion_config = fusion_head_config
    else:
        resolved_fusion_config = FusionHeadConfig.baseline()

    model = PairedBreastModel(
        backbone_name=resolved_backbone_name,
        pretrained=False,
        fusion_head_config=resolved_fusion_config,
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    batches: Iterable[PredictionBatch],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    model.to(device)
    use_cuda = str(device) in ("cuda",) or str(device).startswith("cuda:")
    predictions: dict[str, float] = {}
    for batch in batches:
        breast_ids = batch["breast_id"]
        cc_images = batch["cc_image"].to(device, non_blocking=use_cuda)
        mlo_images = batch["mlo_image"].to(device, non_blocking=use_cuda)
        logits = cast(torch.Tensor, model(cc_images, mlo_images))
        probs = cast(list[float], torch.sigmoid(logits).detach().cpu().tolist())
        for breast_id, prob in zip(breast_ids, probs, strict=True):
            predictions[str(breast_id)] = float(prob)
    return predictions


def _collate_prediction_samples(
    samples: Sequence[PairedBreastSample],
) -> PredictionBatch:
    return {
        "breast_id": [sample["breast_id"] for sample in samples],
        "cc_image": torch.stack([sample["cc_image"] for sample in samples]),
        "mlo_image": torch.stack([sample["mlo_image"] for sample in samples]),
    }
