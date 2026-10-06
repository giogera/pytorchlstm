"""Data loading and preprocessing utilities."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float32]


def create_windows(series: Array, length: int, target: Array | None = None) -> tuple[Array, Array]:
    """Transform a series into (X, y) pairs.

    `series` is either 1D (n,) or 2D (n, n_features).
    X has shape (n - length, length, n_features): `length` consecutive rows.
    y has shape (n - length,): the `target` value that follows each window.

    `target` defaults to the series itself and is required when the series has
    more than one feature.
    """
    if series.ndim == 1:
        series = series[:, np.newaxis]
    elif series.ndim != 2:
        raise ValueError("Series must be 1D or 2D")
    if target is None:
        if series.shape[1] != 1:
            raise ValueError("target is required for a multivariate series")
        target = series[:, 0]
    if target.shape != (len(series),):
        raise ValueError(f"target must have shape ({len(series)},), got {target.shape}")
    if len(series) <= length:
        raise ValueError("Series is too short for the requested length")

    n = len(series) - length
    X = np.stack([series[i : i + length] for i in range(n)])
    y = target[length:]
    return X, y


def column_stats(values: Array) -> tuple[Array, Array]:
    """Per-column mean and std of a 2D array.

    Constant columns get a std of 1 so normalizing them doesn't divide by zero.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> Array:
    """Return the given columns as a float32 array of shape (rows, len(columns))."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    return df[list(columns)].to_numpy(dtype=np.float32)
