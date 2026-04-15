from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image

from final_project.data.dataset import PairedBreastDataset
from final_project.data.manifest import BreastManifestRecord
from final_project.data.preprocess import preprocess_view_image
from final_project.data.transforms import build_image_transform


def _write_uniform_grayscale_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (10, 6), color=value).save(path)


def _steps(transform: object) -> list[Any]:
    return cast(list[Any], cast(Any, transform).transforms)


def _step_names(transform: object) -> list[str]:
    return [type(step).__name__ for step in _steps(transform)]


def test_preprocess_view_image_canonicalizes_laterality_and_sample_shape() -> None:
    image = Image.new("L", (8, 8), color=0)
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), 255 if x < 4 else 64)

    processed = preprocess_view_image(image, breast_id="123_R")
    tensor = cast(
        torch.Tensor,
        build_image_transform(
            image_size=16,
            training=False,
            transform_profile="baseline",
        )(processed),
    )

    assert processed.mode == "RGB"
    assert processed.size == (4, 4)
    assert processed.getpixel((0, 0)) == (64, 64, 64)
    assert processed.getpixel((3, 0)) == (255, 255, 255)
    assert tuple(tensor.shape) == (3, 16, 16)


def test_build_image_transform_baseline_preserves_current_steps() -> None:
    eval_transform = build_image_transform(
        image_size=16,
        training=False,
        transform_profile="baseline",
    )
    train_transform = build_image_transform(
        image_size=16,
        training=True,
        transform_profile="baseline",
    )

    assert _step_names(eval_transform) == ["Resize", "ToTensor"]
    assert _step_names(train_transform) == [
        "Resize",
        "RandomHorizontalFlip",
        "ToTensor",
    ]
    assert cast(Any, _steps(train_transform)[1]).p == 0.0


def test_build_image_transform_normaug_adds_normalize_and_train_flip() -> None:
    eval_transform = build_image_transform(
        image_size=16,
        training=False,
        transform_profile="normaug",
    )
    train_transform = build_image_transform(
        image_size=16,
        training=True,
        transform_profile="normaug",
    )

    assert _step_names(eval_transform) == ["Resize", "ToTensor", "Normalize"]
    assert _step_names(train_transform) == [
        "Resize",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]
    assert cast(Any, _steps(train_transform)[1]).p == 0.5
    normalize = cast(Any, _steps(eval_transform)[-1])
    assert tuple(normalize.mean) == (0.485, 0.456, 0.406)
    assert tuple(normalize.std) == (0.229, 0.224, 0.225)


def test_build_image_transform_normonly_adds_normalize_without_flip() -> None:
    eval_transform = build_image_transform(
        image_size=16,
        training=False,
        transform_profile="normonly",
    )
    train_transform = build_image_transform(
        image_size=16,
        training=True,
        transform_profile="normonly",
    )

    assert _step_names(eval_transform) == ["Resize", "ToTensor", "Normalize"]
    assert _step_names(train_transform) == [
        "Resize",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]
    assert cast(Any, _steps(train_transform)[1]).p == 0.0


def test_paired_breast_dataset_returns_cc_and_mlo_in_fixed_order(
    tmp_path: Path,
) -> None:
    cc_path = tmp_path / "train_img/100_L/100_L_CC.jpg"
    mlo_path = tmp_path / "train_img/100_L/100_L_MLO.jpg"
    _write_uniform_grayscale_image(cc_path, value=24)
    _write_uniform_grayscale_image(mlo_path, value=196)

    dataset = PairedBreastDataset(
        records=[
            BreastManifestRecord(
                breast_id="100_L",
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=1,
            )
        ],
        image_size=12,
        training=False,
        transform_profile="baseline",
    )

    sample = dataset[0]

    assert sample["breast_id"] == "100_L"
    assert sample["label"] is not None
    assert float(sample["label"]) == 1.0
    assert tuple(sample["cc_image"].shape) == (3, 12, 12)
    assert tuple(sample["mlo_image"].shape) == (3, 12, 12)
    assert float(sample["cc_image"].mean()) < float(sample["mlo_image"].mean())


def test_paired_breast_dataset_normaug_normalizes_views(tmp_path: Path) -> None:
    cc_path = tmp_path / "train_img/100_L/100_L_CC.jpg"
    mlo_path = tmp_path / "train_img/100_L/100_L_MLO.jpg"
    _write_uniform_grayscale_image(cc_path, value=24)
    _write_uniform_grayscale_image(mlo_path, value=196)

    dataset = PairedBreastDataset(
        records=[
            BreastManifestRecord(
                breast_id="100_L",
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=1,
            )
        ],
        image_size=12,
        training=False,
        transform_profile="normaug",
    )

    sample = dataset[0]

    expected_cc = (
        (24 / 255.0 - 0.485) / 0.229,
        (24 / 255.0 - 0.456) / 0.224,
        (24 / 255.0 - 0.406) / 0.225,
    )
    expected_mlo = (
        (196 / 255.0 - 0.485) / 0.229,
        (196 / 255.0 - 0.456) / 0.224,
        (196 / 255.0 - 0.406) / 0.225,
    )

    for channel, expected in enumerate(expected_cc):
        assert abs(float(sample["cc_image"][channel, 0, 0]) - expected) < 1e-6
    for channel, expected in enumerate(expected_mlo):
        assert abs(float(sample["mlo_image"][channel, 0, 0]) - expected) < 1e-6


def test_paired_breast_dataset_preprocess_cache_preserves_random_augment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cc_path = tmp_path / "train_img/100_L/100_L_CC.jpg"
    mlo_path = tmp_path / "train_img/100_L/100_L_MLO.jpg"
    _write_uniform_grayscale_image(cc_path, value=24)
    _write_uniform_grayscale_image(mlo_path, value=196)

    dataset = PairedBreastDataset(
        records=[
            BreastManifestRecord(
                breast_id="100_L",
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=1,
            )
        ],
        image_size=12,
        training=True,
        transform_profile="normaug",
        cache_mode="preprocess",
    )

    calls = {"count": 0}

    def fake_transform(image):
        calls["count"] += 1
        return torch.full((3, 12, 12), float(calls["count"]))

    monkeypatch.setattr(dataset, "_transform", fake_transform)

    first = dataset[0]
    second = dataset[0]

    assert float(first["cc_image"][0, 0, 0]) == 1.0
    assert float(second["cc_image"][0, 0, 0]) == 3.0
