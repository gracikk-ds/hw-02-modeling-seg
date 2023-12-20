"""Module to define a custom dataset for the Barcode segmentation task."""

import os
from typing import Dict, Tuple, Union

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

TransformType = Union[albu.BasicTransform, albu.BaseCompose]
DataAnnotation = Union[NDArray[np.uint8], NDArray[np.float32]]


class BarcodeDataset(Dataset):  # type: ignore
    """
    Custom dataset for the Barcode segmaentation task.

    Attributes:
        dataframe (pd.DataFrame): The dataset's metadata including image IDs and labels.
        image_folder (str): Path to the folder containing the images.
        transforms (TransformType): Albumentations transformations to be applied on the images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_folder: str,
        transforms: TransformType,
    ) -> None:
        """
        Initialize a new instance of BarcodeDataset.

        Args:
            dataframe (pd.DataFrame): Dataset's metadata.
            image_folder (str): Path to the folder containing the images.
            transforms (TransformType): Albumentations transformations to apply on the images.
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms

    @staticmethod
    def load_image(image_path: str) -> NDArray[np.uint8]:
        """Load an image from the given path.

        Args:
            image_path (str): Path to the image.

        Returns:
            NDArray[np.uint8]: The loaded image.
        """
        try:
            image: NDArray[np.uint8] = jpeg.JPEG(image_path).decode()
        except RuntimeError:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def create_mask(image_shape: Tuple[int, int], barcode_coords: Tuple[int, int, int, int]) -> NDArray[np.int64]:
        """Create a mask for the barcode in the image.

        Args:
            image_shape (Tuple[int, int]): Shape of the image.
            barcode_coords (Tuple[int, int, int, int]): Coordinates of the barcode in the image.

        Returns:
            NDArray[np.int32]: The created mask.
        """
        mask = np.zeros(image_shape, dtype=np.int64)
        x1, y1, width, height = barcode_coords
        x2, y2 = x1 + width, y1 + height
        mask[y1:y2, x1:x2] = 1
        return mask

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the image and its label based on the provided index.

        Args:
            idx (int): Index of the desired dataset item.

        Returns:
            tuple: A tuple containing the RGB image (np.ndarray) and its labels (np.ndarray).
        """
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row["filename"])
        image = self.load_image(image_path)

        # get image mask
        bbox = (row["x_from"], row["y_from"], row["width"], row["height"])
        mask = self.create_mask(image.shape[:2], bbox)  # type: ignore

        transformed_data: Dict[str, Tensor] = self.transforms(image=image, mask=mask)
        return transformed_data["image"], transformed_data["mask"].to(torch.int64)

    def __len__(self) -> int:
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.dataframe)
