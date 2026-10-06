import numpy as np
import pandas as pd
import pytest

from pytorchlstm.data import column_stats, create_windows, select_columns


def test_series_format():
    """Test create_windows() with a toy series."""
    series = np.arange(10, dtype=np.float32)

    X, y = create_windows(series, 3)

    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert list(X[0, :, 0]) == [0, 1, 2]
    assert y[0] == 3


def test_series_too_short():
    with pytest.raises(ValueError):
        create_windows(np.arange(3, dtype=np.float32), 3)


def test_multivariate_windows():
    series = np.stack([np.arange(10), np.arange(10) * 10], axis=1).astype(np.float32)
    target = np.arange(100, 110, dtype=np.float32)

    X, y = create_windows(series, 3, target)

    assert X.shape == (7, 3, 2)
    assert y.shape == (7,)
    assert X[0].tolist() == [[0, 0], [1, 10], [2, 20]]
    assert y[0] == 103


def test_multivariate_requires_target():
    series = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        create_windows(series, 3)


def test_target_length_must_match():
    series = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        create_windows(series, 3, np.zeros(9, dtype=np.float32))


def test_column_stats_per_column():
    values = np.array([[1, 5], [3, 5]], dtype=np.float32)

    mean, std = column_stats(values)

    assert mean.tolist() == [2, 5]
    assert std.tolist() == [1, 1]  # constant column gets std 1


def test_select_columns_order_and_missing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    assert select_columns(df, ["b", "a"]).tolist() == [[3, 1], [4, 2]]
    with pytest.raises(ValueError, match="c"):
        select_columns(df, ["a", "c"])
