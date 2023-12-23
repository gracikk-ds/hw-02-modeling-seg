"""This module provides image transformation utilities using Albumentations."""

from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

TransformType = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(cfg: DictConfig) -> TransformType:
    """
    Create a composition of image transformations for data augmentation and preprocessing.

    Args:
        cfg (DictConfig): Hydra configuration containing transformation parameters.

    Returns:
        TransformType: An Albumentations composition of image transformations.
    """
    preprocessing = cfg.preprocessing
    augmentations = cfg.augmentations
    postprocessing = cfg.postprocessing

    transforms = []

    if preprocessing:
        transforms.append(
            albu.Resize(height=cfg.height, width=cfg.width),
        )

    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=cfg.flip_probability),
                albu.VerticalFlip(p=cfg.flip_probability),
                albu.HueSaturationValue(
                    hue_shift_limit=cfg.hue_shift_limit,
                    sat_shift_limit=cfg.sat_shift_limit,
                    val_shift_limit=cfg.val_shift_limit,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                ),
                albu.ShiftScaleRotate(),
                albu.GaussianBlur(),
            ],
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
