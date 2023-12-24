"""Utility module for reading dataframes from a specified path based on a given mode."""
import os
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label
from sklearn.model_selection import train_test_split

TEST_VAL_PROPORTION: float = 0.5
RANDOM_STATE: int = 42


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Read the dataframe for the given mode.

    Args:
        data_path (str): Path to the data folder.
        mode (str): Mode specifying which data to read (train, val, or test).

    Returns:
        pd.DataFrame: Dataframe containing data for the specified mode.
    """
    return pd.read_csv(os.path.join(data_path, f"{mode}_data.csv"))


def save_df(dataframe: pd.DataFrame, data_path: str, mode: str) -> None:
    """
    Save the given dataframe to a CSV file.

    Args:
        dataframe (pd.DataFrame): Dataframe to save.
        data_path (str): Path to the data folder.
        mode (str): Mode specifying which data to save (train, val, or test).
    """
    path_to_save = os.path.join(data_path, f"{mode}_data.csv")
    dataframe.to_csv(path_to_save, index=False)


def split_and_save_datasets(data_path: str, train_size: float) -> None:
    """
    Split dataset from given CSV file into training, validation, and test datasets.

    Parameters:
        data_path (str): The path to the input CSV file.
        train_size (float): Proportion of the dataset to include in the training split (0.0 to 1.0).
    """
    dataframe = pd.read_csv(os.path.join(data_path, "annotations.tsv"), sep="\t")

    train_data, temp_data = train_test_split(dataframe, train_size=train_size, random_state=RANDOM_STATE)
    val_data, test_data = train_test_split(temp_data, test_size=TEST_VAL_PROPORTION, random_state=RANDOM_STATE)

    save_df(train_data, data_path, "train")
    save_df(val_data, data_path, "val")
    save_df(test_data, data_path, "test")


def masks_to_bboxes(mask: NDArray[np.uint8]) -> List[List[int]]:
    """
    Convert a binary mask with potentially multiple objects to a list of bounding boxes in COCO format.

    Args:
        mask (NDArray[np.uint8]): A binary mask where objects' pixels are 1 and the background is 0.

    Returns:
        List[List[int]]: A list of bounding boxes, each in the format [x_min, y_min, width, height].
    """
    # Label different components (barcodes)
    labeled_array, num_features = label(mask)

    bboxes = []
    for barcode_label in range(1, num_features + 1):
        # Find the bounding box for each labeled component
        barcode_mask = labeled_array == barcode_label
        rows = np.any(barcode_mask, axis=1)
        cols = np.any(barcode_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        bboxes.append([int(x_min), int(y_min), int(bbox_width), int(bbox_height)])
    return bboxes
