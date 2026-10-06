import numpy as np
import pytest

from pytorchlstm.evaluation import mae, rmse


def test_mae_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([2.0, 2.0, 5.0], dtype=np.float32)

    # absoolute errors: 1, 0, 2 -> mean = 1.0
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_hand_computed():
    y_true = np.array([0.0, 0.0], dtype=np.float32)
    y_pred = np.array([3.0, 4.0], dtype=np.float32)

    # squared errors: 9, 16 -> mean = 12.5 -> square root ~ 3.5355
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
