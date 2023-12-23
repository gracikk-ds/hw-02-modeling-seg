"""
This module provides functionalities to set up evaluation metrics used in a neural network evaluation process.

Functions:
    get_metrics: Generates a collection of essential evaluation metrics.
"""
from torchmetrics import MetricCollection, PearsonCorrCoef
from torchmetrics.retrieval import RetrievalNormalizedDCG


def get_metrics(**kwargs) -> MetricCollection:
    """Generate a collection of essential evaluation metrics.

    This function creates a collection of metrics including Pearson, MAE, and nDCG using the provided keyword arguments
    for their initialization.

    Args:
        kwargs: Arbitrary keyword arguments that are forwarded to the initialization of each metric.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            "Pearson": PearsonCorrCoef(**kwargs),
            "nDCG@100": RetrievalNormalizedDCG(top_k=100, **kwargs),
            "nDCG@1000": RetrievalNormalizedDCG(top_k=1000, **kwargs),
        },
    )
