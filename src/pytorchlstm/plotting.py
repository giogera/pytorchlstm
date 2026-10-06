"""Plotting utilities."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

Array = npt.NDArray[np.float32]


def _check_same_length(y_true: Array, y_pred: Array) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")


def plot_loss(
    history: Sequence[float],
    val_history: Sequence[float] | None = None,
    best_epoch: int | None = None,
    log_scale: bool = False,
) -> Figure:
    """Training loss per epoch, optionally with validation loss."""
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.add_subplot()
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, marker="o", markersize=3, label="Train")
    if val_history is not None:
        ax.plot(
            range(1, len(val_history) + 1),
            val_history,
            marker="o",
            markersize=3,
            label="Validation",
        )
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss per epoch")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_predictions(y_true: Array, y_pred: Array, baseline: Array | None = None) -> Figure:
    """Actual vs predicted values over the test period."""
    _check_same_length(y_true, y_pred)
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(y_true, label="Actual", color="black", linewidth=1.5)
    ax.plot(y_pred, label="LSTM", color="tab:red", linewidth=1.2)
    if baseline is not None:
        _check_same_length(y_true, baseline)
        ax.plot(
            baseline,
            label="Baseline (last value)",
            color="tab:blue",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Test step")
    ax.set_ylabel("Value")
    ax.set_title("Predictions on the test set")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_scatter(y_true: Array, y_pred: Array) -> Figure:
    """Predicted vs actual: a perfect model lies on the diagonal."""
    _check_same_length(y_true, y_pred)
    fig = Figure(figsize=(5, 5), layout="constrained")
    ax = fig.add_subplot()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    return fig


def plot_residuals(y_true: Array, y_pred: Array) -> Figure:
    """Residuals over time and their distribution."""
    _check_same_length(y_true, y_pred)
    residuals = y_true - y_pred
    fig = Figure(figsize=(10, 4), layout="constrained")

    ax_time = fig.add_subplot(1, 2, 1)
    ax_time.plot(residuals, linewidth=1)
    ax_time.axhline(0.0, color="black", linewidth=1)
    ax_time.set_xlabel("Test step")
    ax_time.set_ylabel("Actual - predicted")
    ax_time.set_title("Residuals over time")
    ax_time.grid(alpha=0.3)

    ax_hist = fig.add_subplot(1, 2, 2)
    spread = float(residuals.max() - residuals.min())
    n_bins = 30 if spread > 1e-6 else 1
    ax_hist.hist(residuals, bins=n_bins)
    ax_hist.axvline(0.0, color="black", linewidth=1)
    ax_hist.set_xlabel("Actual - predicted")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Residual distribution")
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
