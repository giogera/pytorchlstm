from dataclasses import asdict

import numpy as np
import pytest
import torch

from pytorchlstm.checkpoint import load_checkpoint, save_checkpoint
from pytorchlstm.config import Config
from pytorchlstm.model import LSTMModel
from pytorchlstm.training import predict


def test_predictions_identical_after_reload(tmp_path):
    config = Config.from_dict({"model": {"hidden": 8}})
    model = LSTMModel(hidden=8)
    X = np.random.default_rng(0).standard_normal((5, 10, 1)).astype(np.float32)
    before = predict(model, X)
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, config, mean=1.5, std=2.5)
    loaded = load_checkpoint(path)

    np.testing.assert_allclose(predict(loaded.model, X), before)


def test_config_and_statistics_are_preserved(tmp_path):
    config = Config.from_dict({"model": {"hidden": 8}, "data": {"window": 12}})
    path = tmp_path / "model.pt"

    save_checkpoint(path, LSTMModel(hidden=8), config, mean=1.5, std=2.5)
    loaded = load_checkpoint(path)

    assert loaded.config == config
    assert loaded.mean == [1.5]
    assert loaded.std == [2.5]
    assert loaded.target_mean == 1.5
    assert loaded.target_std == 2.5


def test_multivariate_statistics_are_preserved(tmp_path):
    config = Config.from_dict(
        {"model": {"hidden": 8, "n_features": 2}, "data": {"features": ["a", "b"]}}
    )
    model = LSTMModel(n_features=2, hidden=8)
    X = np.random.default_rng(0).standard_normal((5, 10, 2)).astype(np.float32)
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, config, [1.0, 2.0], [3.0, 4.0], 5.0, 6.0)
    loaded = load_checkpoint(path)

    assert loaded.config.data.feature_columns == ["a", "b"]
    assert loaded.mean == [1.0, 2.0]
    assert loaded.std == [3.0, 4.0]
    assert (loaded.target_mean, loaded.target_std) == (5.0, 6.0)
    np.testing.assert_allclose(predict(loaded.model, X), predict(model, X))


def test_target_stats_default_to_matching_feature(tmp_path):
    config = Config.from_dict(
        {"model": {"n_features": 2}, "data": {"target": "b", "features": ["a", "b"]}}
    )
    path = tmp_path / "model.pt"

    save_checkpoint(path, LSTMModel(n_features=2), config, [1.0, 2.0], [3.0, 4.0])
    loaded = load_checkpoint(path)

    assert (loaded.target_mean, loaded.target_std) == (2.0, 4.0)


def test_target_stats_required_when_target_not_a_feature(tmp_path):
    config = Config.from_dict(
        {"model": {"n_features": 2}, "data": {"target": "c", "features": ["a", "b"]}}
    )

    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "model.pt", LSTMModel(n_features=2), config, [0, 0], [1, 1])


def test_loads_legacy_univariate_checkpoint(tmp_path):
    """Checkpoints saved before multivariate support store scalar statistics."""
    config = Config.from_dict({"model": {"hidden": 8}})
    model = LSTMModel(hidden=8)
    raw_config = asdict(config)
    del raw_config["data"]["target"], raw_config["data"]["features"]
    path = tmp_path / "old.pt"
    torch.save(
        {"state_dict": model.state_dict(), "config": raw_config, "mean": 1.5, "std": 2.5},
        path,
    )

    loaded = load_checkpoint(path)

    assert loaded.mean == [1.5]
    assert (loaded.target_mean, loaded.target_std) == (1.5, 2.5)
