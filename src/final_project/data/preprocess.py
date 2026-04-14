from __future__ import annotations

from PIL import Image


def preprocess_view_image(image: Image.Image, breast_id: str) -> Image.Image:
    processed = _crop_black_borders(image)
    processed = _canonicalize_laterality(processed, breast_id)
    return processed.convert("RGB")


def _crop_black_borders(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    bbox = grayscale.getbbox()
    if bbox is None:
        return image.copy()
    return image.crop(bbox)


def _canonicalize_laterality(image: Image.Image, breast_id: str) -> Image.Image:
    if breast_id.endswith("_R"):
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image.copy()
