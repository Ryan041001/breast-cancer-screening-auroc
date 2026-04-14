from __future__ import annotations

from pathlib import Path

from PIL import Image

from final_project.data.dataset import PairedBreastDataset
from final_project.data.manifest import BreastManifestRecord
from final_project.data.preprocess import preprocess_view_image
from final_project.data.transforms import build_image_transform


def _write_uniform_grayscale_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (10, 6), color=value).save(path)


def test_preprocess_view_image_canonicalizes_laterality_and_sample_shape() -> None:
    image = Image.new("L", (8, 8), color=0)
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), 255 if x < 4 else 64)

    processed = preprocess_view_image(image, breast_id="123_R")
    tensor = build_image_transform(image_size=16, training=False)(processed)

    assert processed.mode == "RGB"
    assert processed.size == (4, 4)
    assert processed.getpixel((0, 0)) == (64, 64, 64)
    assert processed.getpixel((3, 0)) == (255, 255, 255)
    assert tuple(tensor.shape) == (3, 16, 16)


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
    )

    sample = dataset[0]

    assert sample["breast_id"] == "100_L"
    assert float(sample["label"]) == 1.0
    assert tuple(sample["cc_image"].shape) == (3, 12, 12)
    assert tuple(sample["mlo_image"].shape) == (3, 12, 12)
    assert float(sample["cc_image"].mean()) < float(sample["mlo_image"].mean())
