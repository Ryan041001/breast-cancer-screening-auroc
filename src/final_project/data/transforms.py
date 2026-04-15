from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from PIL import Image
from torchvision import transforms


TransformProfile = Literal["baseline", "normaug", "normonly"]


def build_image_transform(
    image_size: int,
    training: bool,
    transform_profile: TransformProfile = "baseline",
) -> Callable[[Image.Image], torch.Tensor]:
    steps: list[object] = [
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        flip_probability = 0.5 if transform_profile == "normaug" else 0.0
        steps.append(transforms.RandomHorizontalFlip(p=flip_probability))
    steps.append(transforms.ToTensor())
    if transform_profile in {"normaug", "normonly"}:
        steps.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms.Compose(steps)
