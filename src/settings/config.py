"""
This module provides configuration models for training and data transformation setups.

It contains models for:
- Loss configurations, defining different losses and their properties.
- Data transformation configurations, detailing various augmentation and preprocessing steps.
- Data loading configurations, specifying dataset paths and related properties.
- The main configuration model, bringing together all the aforementioned configurations for a cohesive training setup.
"""

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """
    Configuration for loss functions.

    Attributes:
        name (str): Name of the loss.
        weight (float): Weight of the loss.
        loss_fn (str): Loss function.
        loss_kwargs (Dict[str, Any]): Additional keyword arguments for the loss function.
    """

    name: str
    weight: float
    loss_fn: str
    loss_kwargs: Dict[str, Any]


class TransformsConfig(BaseModel):
    """
    Configuration for data transformations.

    Attributes:
        width (int): Image width after transformation.
        height (int): Image height after transformation.
        preprocessing (bool): Whether to apply preprocessing.
        augmentations (bool): Whether to apply augmentations.
        postprocessing (bool): Whether to apply postprocessing.
        flip_probability (float): Probability of applying flipping augmentation.
        brightness_limit (float): Limit for brightness augmentation.
        contrast_limit (float): Limit for contrast augmentation.
        hue_shift_limit (int): Limit for hue shift augmentation.
        sat_shift_limit (int): Limit for saturation shift augmentation.
        val_shift_limit (int): Limit for value shift augmentation.
        blur_probability (float): Probability of applying blur augmentation.
    """

    width: int = 224
    height: int = 224
    preprocessing: bool = True
    augmentations: bool = True
    postprocessing: bool = True
    flip_probability: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 20
    sat_shift_limit: int = 30
    val_shift_limit: int = 20
    blur_probability: float = 0.5


class DataConfig(BaseModel):
    """
    Configuration for data loading.

    Attributes:
        data_path (str): Path to the dataset.
        batch_size (int): Number of samples per batch.
        n_workers (int): Number of worker processes for data loading.
        train_size (float): Proportion of the dataset used for training.
    """

    data_path: str
    batch_size: int
    num_workers: Optional[int]
    train_size: float


class Config(BaseModel):
    """
    Main configuration class for the project.

    Attributes:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
        data_config (DataConfig): Data loading configuration.
        transforms_config (TransformsConfig): Data transformation configuration.
        n_epochs (int): Number of training epochs.
        num_classes (int): Number of classes in the dataset.
        accelerator (str): Type of accelerator (e.g., "CPU", "GPU").
        monitor_metric (str): Metric to monitor during training.
        monitor_mode (str): Mode for monitoring the metric (e.g., "min", "max").
        net_kwargs (Dict[str, Any]): Additional keyword arguments for the model.
        optimizer (str): Optimizer for training.
        optimizer_kwargs (Dict[str, Any]): Additional keyword arguments for the optimizer.
        scheduler (str): Learning rate scheduler.
        scheduler_kwargs (Dict[str, Any]): Additional keyword arguments for the scheduler.
        losses (List[LossConfig]): List of loss configurations.
        progress_bar_refresh_rate (int): progress bar refresh rate for Lightning callback
        precision (int): Training precision
    """

    project_name: str
    experiment_name: str
    base_data_settings: DataConfig
    transforms_settings: TransformsConfig
    max_steps: int
    num_classes: int
    accelerator: str
    monitor_metric: str
    monitor_mode: str
    net_kwargs: Dict[str, Any]
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    scheduler: str
    scheduler_kwargs: Dict[str, Any]
    losses: List[LossConfig]
    progress_bar_refresh_rate: int
    precision: int = 16
    dotenv_path: str = ".env"

    @classmethod
    def from_yaml(cls, path: str) -> DictConfig:
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            DictConfig: An instance of the Config class.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)  # type: ignore
