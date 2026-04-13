# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from torchvision import transforms


def get_image_augmentation(
    color_jitter: bool = True,
    gray_scale: bool = True,
    gau_blur: bool = False,
    img_norm: bool = False,
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Whether to apply color jitter (default: True)
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)
        img_norm: Whether to apply image normalization (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    if color_jitter:
        transform_list.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=default_jitter["brightness"],
                        contrast=default_jitter["contrast"],
                        saturation=default_jitter["saturation"],
                        hue=default_jitter["hue"],
                    )
                ],
                p=default_jitter["p"],
            )
        )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    if img_norm:
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    return transforms.Compose(transform_list) if transform_list else None

