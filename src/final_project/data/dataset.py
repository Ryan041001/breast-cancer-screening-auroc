from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import Dataset

from final_project.data.manifest import BreastManifestRecord
from final_project.data.preprocess import preprocess_view_image
from final_project.data.transforms import TransformProfile, build_image_transform


class PairedBreastSample(TypedDict):
    breast_id: str
    cc_image: torch.Tensor
    mlo_image: torch.Tensor
    label: torch.Tensor | None


class PairedBreastDataset(Dataset[PairedBreastSample]):
    def __init__(
        self,
        records: Sequence[BreastManifestRecord],
        image_size: int,
        training: bool,
        transform_profile: TransformProfile = "baseline",
    ) -> None:
        self._records = list(records)
        self._transform = build_image_transform(
            image_size=image_size,
            training=training,
            transform_profile=transform_profile,
        )
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_cache"):
            self._cache = {}

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PairedBreastSample:
        record = self._records[index]
        cc_image = self._load_view(record.cc_path, record.breast_id)
        mlo_image = self._load_view(record.mlo_path, record.breast_id)
        label = None if record.label is None else torch.tensor(float(record.label))
        return {
            "breast_id": record.breast_id,
            "cc_image": cc_image,
            "mlo_image": mlo_image,
            "label": label,
        }

    def _load_view(self, path: Path, breast_id: str) -> torch.Tensor:
        cache_key = (str(path), breast_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with Image.open(path) as image:
            processed = preprocess_view_image(image, breast_id=breast_id)
        tensor = self._transform(processed)
        self._cache[cache_key] = tensor
        return tensor
