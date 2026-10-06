import numpy as np
import pytest

from pytorchlstm.data import create_windows
from pytorchlstm.evaluation import mae
from pytorchlstm.model import LSTMModel
from pytorchlstm.training import fit, predict, set_seed, train


@pytest.fixture
def sine_data():
    series = np.sin(np.linspace(0, 20, 200)).astype(np.float32)
    return create_windows(series, 10)


def test_loss_decreases(sine_data):
    X, y = sine_data
    set_seed(0)
    history = train(LSTMModel(hidden=8), X, y, epochs=10)
    assert history[-1] < history[0]


def test_training_is_reproducible(sine_data):
    X, y = sine_data

    def run() -> list[float]:
        set_seed(0)
        return train(LSTMModel(hidden=8), X, y, epochs=3)

    assert run() == pytest.approx(run())


def test_predict_shape(sine_data):
    X, y = sine_data

    preds = predict(LSTMModel(hidden=8), X)

    assert preds.shape == y.shape


def test_model_beats_constant_prediction(sine_data):
    X, y = sine_data
    set_seed(0)
    model = LSTMModel(hidden=16)

    train(model, X, y, epochs=30)
    preds = predict(model, X)

    constant = np.full_like(y, y.mean())
    assert mae(y, preds) < mae(y, constant)


def test_fit_returns_consistent_history(sine_data):
    X, y = sine_data

    result = fit(LSTMModel(hidden=8), X[:150], y[:150], X[150:], y[150:], epochs=5)

    assert len(result.train_loss) == len(result.val_loss)
    assert 1 <= result.best_epoch <= len(result.val_loss)


def test_fit_restores_best_weights(sine_data):
    X, y = sine_data
    model = LSTMModel(hidden=8)

    result = fit(model, X[:150], y[:150], X[150:], y[150:], epochs=8, patience=8)

    preds = predict(model, X[150:])
    final_val = float(np.mean((preds - y[150:]) ** 2))
    assert final_val == pytest.approx(min(result.val_loss), rel=1e-4)


def test_fit_stops_early_on_noise():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10, 1)).astype(np.float32)
    y = rng.standard_normal(200).astype(np.float32)  # unlearnable target

    result = fit(LSTMModel(hidden=16), X[:140], y[:140], X[140:], y[140:], epochs=100, patience=3)

    assert result.stopped_early
    assert len(result.train_loss) < 100


def test_fit_rejects_invalid_patience(sine_data):
    X, y = sine_data

    with pytest.raises(ValueError):
        fit(LSTMModel(hidden=8), X[:150], y[:150], X[150:], y[150:], patience=0)
