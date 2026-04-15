from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocess import preprocess_view_image
from .transforms import TransformProfile, build_image_transform


POSITIVE_PATHOLOGIES = frozenset({"malignant"})
NEGATIVE_PATHOLOGIES = frozenset({"benign", "normal"})
FALLBACK_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True, slots=True)
class ExternalImageRecord:
    sample_id: str
    image_path: Path
    label: int
    dataset: str
    patient_id: str
    laterality: str
    view: str
    pathology: str


class ExternalImageSample(TypedDict):
    sample_id: str
    image: torch.Tensor
    label: torch.Tensor


def load_external_split_records(
    *,
    catalog_csv: str | Path,
    processed_root: str | Path,
    split_csv: str | Path,
    max_samples: int | None = None,
) -> list[ExternalImageRecord]:
    processed_root_path = Path(processed_root).resolve(strict=False)
    allowed_keys = {
        (
            _require_value(row, "dataset"),
            _require_value(row, "patient_id"),
            _require_value(row, "processed_path"),
        )
        for row in _read_csv_rows(Path(split_csv))
    }

    records: list[ExternalImageRecord] = []
    for row in _read_csv_rows(Path(catalog_csv)):
        key = (
            _require_value(row, "dataset"),
            _require_value(row, "patient_id"),
            _require_value(row, "processed_path"),
        )
        if key not in allowed_keys:
            continue
        pathology = _require_value(row, "pathology").strip().lower()
        if pathology in POSITIVE_PATHOLOGIES:
            label = 1
        elif pathology in NEGATIVE_PATHOLOGIES:
            label = 0
        else:
            continue

        dataset = key[0]
        patient_id = key[1]
        processed_path = _resolve_relative_path(
            processed_root_path,
            _require_value(row, "processed_path"),
        )
        if not processed_path.is_file():
            processed_path = _resolve_with_suffix_fallback(processed_path)
        if not processed_path.is_file():
            continue
        laterality = row.get("laterality", "U") or "U"
        view = row.get("view", "U") or "U"
        sample_id = (
            f"{dataset}:{patient_id}:{laterality}:{view}:{processed_path.stem}"
        )
        records.append(
            ExternalImageRecord(
                sample_id=sample_id,
                image_path=processed_path,
                label=label,
                dataset=dataset,
                patient_id=patient_id,
                laterality=laterality,
                view=view,
                pathology=pathology,
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records


class ExternalImageDataset(Dataset[ExternalImageSample]):
    def __init__(
        self,
        records: Sequence[ExternalImageRecord],
        *,
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

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> ExternalImageSample:
        record = self._records[index]
        pseudo_breast_id = "external_R" if record.laterality == "R" else "external_L"
        with Image.open(record.image_path) as image:
            processed = preprocess_view_image(image, breast_id=pseudo_breast_id)
        return {
            "sample_id": record.sample_id,
            "image": self._transform(processed),
            "label": torch.tensor(float(record.label)),
        }


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_value(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required column '{key}'")
    return value


def _resolve_relative_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve(strict=False)


def _resolve_with_suffix_fallback(candidate: Path) -> Path:
    for suffix in FALLBACK_IMAGE_SUFFIXES:
        alternative = candidate.with_suffix(suffix)
        if alternative.is_file():
            return alternative
    return candidate
