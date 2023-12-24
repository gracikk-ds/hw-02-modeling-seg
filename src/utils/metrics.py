"""
This module provides functionalities to set up evaluation metrics used in a neural network evaluation process.

Functions:
    get_metrics: Generates a collection of essential evaluation metrics.
"""
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall


def get_metrics(**kwargs) -> MetricCollection:
    """Generate a collection of essential evaluation metrics.

    This function creates a collection of metrics including IoU, Precision, Recall and F1sscore using the provided
    keyword arguments for their initialization.

    Args:
        kwargs: Arbitrary keyword arguments that are forwarded to the initialization of each metric.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            "IoU": BinaryJaccardIndex(**kwargs),
            "BinaryPrecision": BinaryPrecision(**kwargs),
            "BinaryRecall": BinaryRecall(**kwargs),
            "BinaryF1": BinaryF1Score(**kwargs),
            # TODO: Add detection metrics, count the number of recognized objects.
        },
    )
