"""
This module provides configuration models for training and data transformation setups.

It contains models for:
- Loss configurations, defining different losses and their properties.
- Data transformation configurations, detailing various augmentation and preprocessing steps.
- Data loading configurations, specifying dataset paths and related properties.
- The main configuration model, bringing together all the aforementioned configurations for a cohesive training setup.
"""

from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class GeneralSettings(BaseModel):
    """
    General train settings.

    Attributes:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
        max_steps (int): Number of training steps.
        dotenv_path (str): dotenv path.
    """

    project_name: str
    experiment_name: str
    max_steps: int
    dotenv_path: str = ".env"


class ModelConfig(BaseModel):
    """
    Configuration for Highlight Detector.

    Attributes:
        head_name (str): Name of the head to use.
        encoder_name (str): Name of the encoder to use.
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
    """

    head_name: str
    encoder_name: str
    in_channels: int
    num_classes: int


class HardwareConfiguration(BaseModel):
    """Hardware configuration settings.

    Attributes:
        accelerator (str): Type of accelerator (e.g., "CPU", "GPU").
        devices (Union[List[int], int, str]): Which device to use if there many. Or just use "auto".
        precision (int): Training precision.
    """

    accelerator: str
    devices: Union[List[int], int, str]
    precision: int = 32


class CallbacksConfiguration(BaseModel):
    """Callbacks Configuration settings.

    Attributes:
        monitor_metric (str): Metric to monitor during training.
        monitor_mode (str): Mode for monitoring the metric (e.g., "min", "max").
        early_stopping_patience (int): early stopping patience steps.
        progress_bar_refresh_rate (int): progress bar refresh rate for Lightning callback.
    """

    monitor_metric: str
    monitor_mode: str
    early_stopping_patience: int
    progress_bar_refresh_rate: int


class LossConfig(BaseModel):
    """
    Configuration for loss functions.

    Attributes:
        name (str): Name of the loss.
        loss_weight (float): Weight of the loss.
        loss_fn (str): Loss function.
        loss_kwargs (Dict[str, Any]): Additional keyword arguments for the loss function.
    """

    name: str
    loss_weight: float
    loss_fn: str
    loss_kwargs: Dict[str, Any]


class TransformsConfig(BaseModel):
    """
    Configuration for data transformations.

    Attributes:
        max_size (int): Max image dim after transformation.
        preprocessing (bool): Whether to apply preprocessing.
        augmentations (bool): Whether to apply augmentations.
        flip_probability (float): Probability of applying flipping augmentation.
        brightness_limit (float): Limit for brightness augmentation.
        contrast_limit (float): Limit for contrast augmentation.
        hue_shift_limit (int): Limit for hue shift augmentation.
        sat_shift_limit (int): Limit for saturation shift augmentation.
        val_shift_limit (int): Limit for value shift augmentation.
        blur_probability (float): Probability of applying blur augmentation.
    """

    max_size: int = 256
    preprocessing: bool = True
    augmentations: bool = True
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
        num_workers (int): Number of worker processes for data loading.
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
        general (str): General project settings.
        hardware (HardwareConfiguration): Hardware configuration settings.
        callbacks (CallbacksConfiguration): Callbacks configuration settings.
        base_data_settings (DataConfig): Data loading configuration.
        transforms_settings (TransformsConfig): Data transformation configuration.
        model (ModelConfig): Model configuration.
        losses (List[LossConfig]): List of loss configurations.
        optimizer (str): Optimizer for training.
        optimizer_kwargs (Dict[str, Any]): Additional keyword arguments for the optimizer.
        scheduler (str): Learning rate scheduler.
        scheduler_kwargs (Dict[str, Any]): Additional keyword arguments for the scheduler.

    """

    general: GeneralSettings
    hardware: HardwareConfiguration
    callbacks: CallbacksConfiguration
    base_data_settings: DataConfig
    transforms_settings: TransformsConfig
    model: ModelConfig
    losses: List[LossConfig]
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    scheduler: str
    scheduler_kwargs: Dict[str, Any]

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
