"""
Module to define a custom dataset for the Planet challenge.

This module provides the PlanetDataset class which represents
the dataset of the Planet challenge images and their labels.
It uses the albumentations library for image transformations.
"""

import os
from typing import Dict, Tuple, Union

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

TransformType = Union[albu.BasicTransform, albu.BaseCompose]
DataAnnotation = Union[NDArray[np.uint8], NDArray[np.float32]]


class PlanetDataset(Dataset):  # type: ignore
    """
    Custom dataset for the Planet challenge.

    Attributes:
        dataframe (pd.DataFrame): The dataset's metadata including image IDs and labels.
        image_folder (str): Path to the folder containing the images.
        transforms (TransformType, optional): Albumentations transformations to be applied on the images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_folder: str,
        transforms: TransformType,
    ) -> None:
        """
        Initialize a new instance of PlanetDataset.

        Args:
            dataframe (pd.DataFrame): Dataset's metadata including image IDs and labels.
            image_folder (str): Path to the folder containing the images.
            transforms (TransformType, optional): Albumentations transformations to apply on the images.
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the image and its label based on the provided index.

        Args:
            idx (int): Index of the desired dataset item.

        Returns:
            tuple: A tuple containing the RGB image (np.ndarray) and its labels (np.ndarray).
        """
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, f"{row.image_name}.jpg")
        labels: NDArray[np.float32] = np.array(row.drop(["image_name", "tags"]), dtype="float32")

        try:
            image: NDArray[np.uint8] = jpeg.JPEG(image_path).decode()
        except RuntimeError:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_data: Dict[str, DataAnnotation] = {"image": image, "labels": labels}

        transformed_data: Dict[str, Tensor] = self.transforms(**original_data)

        return transformed_data["image"], transformed_data["labels"]

    def __len__(self) -> int:
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.dataframe)
