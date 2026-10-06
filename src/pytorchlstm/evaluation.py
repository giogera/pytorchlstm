"""Evaluation metrics."""

import numpy as np
import numpy.typing as npt


def mae(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
