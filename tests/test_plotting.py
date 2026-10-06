import numpy as np
import pytest
from matplotlib.figure import Figure

from pytorchlstm.plotting import (
    plot_loss,
    plot_predictions,
    plot_residuals,
    plot_scatter,
    save_figure,
)


@pytest.fixture
def values():
    y_true = np.linspace(0, 1, 50, dtype=np.float32)
    y_pred = y_true + np.float32(0.05)
    return y_true, y_pred


def test_plot_loss_basic():
    fig = plot_loss([1.0, 0.5, 0.25])

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 1


def test_plot_loss_log_scale():
    fig = plot_loss([1.0, 0.5, 0.25], log_scale=True)

    assert fig.axes[0].get_yscale() == "log"


def test_plot_predictions_with_baseline(values):
    y_true, y_pred = values

    fig = plot_predictions(y_true, y_pred, baseline=y_true)

    assert len(fig.axes[0].get_lines()) == 3


def test_plot_predictions_without_baseline(values):
    y_true, y_pred = values

    fig = plot_predictions(y_true, y_pred)

    assert len(fig.axes[0].get_lines()) == 2


def test_plot_predictions_length_mismatch(values):
    y_true, y_pred = values

    with pytest.raises(ValueError):
        plot_predictions(y_true, y_pred[:-1])


def test_plot_scatter(values):
    y_true, y_pred = values

    fig = plot_scatter(y_true, y_pred)

    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[0].get_lines()) == 1


def test_plot_residuals_has_two_panels(values):
    y_true, y_pred = values

    fig = plot_residuals(y_true, y_pred)

    assert len(fig.axes) == 2


def test_save_figure_creates_file(tmp_path):
    fig = plot_loss([1.0, 0.5])

    path = save_figure(fig, tmp_path / "nested" / "loss.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_residuals_constant_error():
    y_true = np.linspace(0, 1, 50, dtype=np.float32)

    fig = plot_residuals(y_true, y_true + np.float32(0.05))

    assert len(fig.axes) == 2


def test_plot_residuals_perfect_prediction():
    y = np.linspace(0, 1, 50, dtype=np.float32)

    fig = plot_residuals(y, y)

    assert len(fig.axes) == 2


def test_plot_loss_with_validation_and_best_epoch():
    fig = plot_loss([1.0, 0.5, 0.4], val_history=[1.1, 0.6, 0.7], best_epoch=2)

    assert len(fig.axes[0].get_lines()) == 3  # train, validation, best epoch
