# pylint:disable=arguments-differ,unused-argument
"""
BarcodeRunner Module.

This module contains the `BarcodeRunner` class, a subclass of `pytorch_lightning.LightningModule`, specifically designed
for segmentation tasks. The runner handles model initialization, metric computation, optimizer and scheduler
configuration, and the main training, validation, and testing loops.
"""

from typing import Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig

from src.utils.general import load_object
from src.utils.losses import get_losses
from src.utils.metrics import get_metrics


class BarcodeRunner(pl.LightningModule):
    """The main LightningModule for image segmentation tasks.

    Attributes:
        config (DictConfig): Configuration object with parameters for model, optimizer, scheduler, etc.
        model (torch.nn.Module): The image segmentation model.
        valid_metrics (pytorch_lightning.Metric): Metrics to track during validation.
        test_metrics (pytorch_lightning.Metric): Metrics to track during testing.
        losses (list): List of loss functions.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the BarcodeRunner with a given configuration.

        Args:
            config (DictConfig): Configuration object containing necessary parameters.
        """
        super().__init__()
        self.config = config

        self._init_model()
        self._init_metrics()
        self.losses = get_losses(self.config.losses)

        self.save_hyperparameters()

    def _init_model(self) -> None:
        """Initialize the model using timm's `create_model` method."""
        self.model = smp.create_model(
            arch=self.config.model.head_name,
            encoder_name=self.config.model.encoder_name,
            in_channels=self.config.model.in_channels,
            classes=self.config.model.num_classes,
        )

    def _init_metrics(self) -> None:
        """Initialize metrics for validation and testing."""
        metrics = get_metrics()
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        optimizer = load_object(self.config.optimizer)(
            self.model.parameters(),
            **self.config.optimizer_kwargs,
        )

        scheduler = load_object(self.config.scheduler)(optimizer, **self.config.scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.callbacks.monitor_metric,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Model's output tensor.
        """
        return self.model(images)

    def _calculate_loss(self, pr_logits: torch.Tensor, gt_labels: torch.Tensor, prefix: str) -> torch.Tensor:
        """
        Calculate the total loss and logs individual and total losses.

        Args:
            pr_logits (torch.Tensor): Predicted logits from the model.
            gt_labels (torch.Tensor): Ground truth labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            torch.Tensor: Computed total loss.
        """
        class_loss = self.losses["ClassificationLoss"].loss(pr_logits, gt_labels)  # type: ignore
        seg_loss = self.losses["SegmentationLoss"].loss(pr_logits, gt_labels)  # type: ignore
        total_loss = (
            self.losses["ClassificationLoss"].loss_weight * class_loss
            + self.losses["SegmentationLoss"].loss_weight * seg_loss
        )
        self.log(f"{prefix}ClassificationLoss", class_loss.item(), sync_dist=True)
        self.log(f"{prefix}SegmentationLoss", seg_loss.item(), sync_dist=True)
        self.log(f"{prefix}TotalLoss", total_loss.item(), sync_dist=True)

        return total_loss

    def _process_batch(self, batch, prefix: str) -> Optional[torch.Tensor]:
        """
        Process a batch of images and labels for either training, validation, or testing.

        Args:
            batch (tuple): A tuple containing images and ground truth labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            Optional[torch.Tensor]: Computed total loss for train step.
        """
        images, gt_labels = batch
        pr_logits = self(images)

        if "train" in prefix:
            return self._calculate_loss(pr_logits, gt_labels, prefix)

        self._calculate_loss(pr_logits, gt_labels, prefix)
        probs = torch.sigmoid(pr_logits)

        if "val" in prefix:
            self.valid_metrics(probs[:, 1, ...], gt_labels)
        elif "test" in prefix:
            self.test_metrics(probs[:, 1, ...], gt_labels)
        return None

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Process a batch during training.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed Loss

        """
        return self._process_batch(batch, "train_")  # type: ignore

    def validation_step(self, batch, batch_idx) -> None:
        """
        Process a batch during validation.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "val_")

    def test_step(self, batch, batch_idx) -> None:
        """
        Process a batch during testing.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "test_")

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a validation epoch."""
        self.valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log the computed validation metrics at the end of a validation epoch."""
        self.log_dict(self.valid_metrics.compute(), on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Log the computed test metrics at the end of a testing epoch."""
        self.log_dict(self.test_metrics.compute(), on_epoch=True, sync_dist=True)
