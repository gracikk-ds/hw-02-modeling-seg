# pylint:disable=arguments-differ,unused-argument
"""
PlanetRunner Module.

This module contains the `PlanetRunner` class, a subclass of
`pytorch_lightning.LightningModule`, specifically designed for multilabel image
classification tasks. The runner handles model initialization, metric computation,
optimizer and scheduler configuration, and the main training, validation, and testing loops.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from timm import create_model

from src.utils.general import load_object
from src.utils.losses import get_losses
from src.utils.metrics import get_metrics


class PlanetRunner(pl.LightningModule):
    """The main LightningModule for multilabel image classification tasks.

    Attributes:
        config (DictConfig): Configuration object with parameters for model, optimizer, scheduler, etc.
        model (torch.nn.Module): The image classification model.
        valid_metrics (pytorch_lightning.Metric): Metrics to track during validation.
        test_metrics (pytorch_lightning.Metric): Metrics to track during testing.
        losses (list): List of loss functions.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the PlanetRunner with a given configuration.

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
        self.model = create_model(num_classes=self.config.num_classes, **self.config.net_kwargs)

    def _init_metrics(self) -> None:
        """Initialize metrics for validation and testing."""
        metrics = get_metrics(
            num_classes=self.config.num_classes,
            num_labels=self.config.num_classes,
            task="multilabel",
            average="macro",
            threshold=0.5,
        )
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
                "monitor": self.config.monitor_metric,
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
        total_loss: torch.Tensor = torch.tensor(0, dtype=torch.float32).to(pr_logits.device)
        for cur_loss in self.losses:
            loss_value: torch.Tensor = cur_loss.loss(pr_logits, gt_labels)
            weighted_loss_value = cur_loss.weight * loss_value
            total_loss += weighted_loss_value
            self.log(
                f"{prefix}{cur_loss.name}_loss",
                loss_value.item(),
                sync_dist=True,
            )

        self.log(f"{prefix}total_loss", total_loss.item(), sync_dist=True)

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
        pr_labels = torch.sigmoid(pr_logits)

        if "val" in prefix:
            self.valid_metrics(pr_labels, gt_labels)
        elif "test" in prefix:
            self.test_metrics(pr_labels, gt_labels)
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
