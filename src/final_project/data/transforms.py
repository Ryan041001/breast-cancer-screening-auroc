from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms


def build_image_transform(
    image_size: int,
    training: bool,
) -> Callable[[Image.Image], torch.Tensor]:
    steps: list[object] = [
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        steps.append(transforms.RandomHorizontalFlip(p=0.0))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
