import json

import pytest

from pytorchlstm.config import Config, load_config, save_config


def test_defaults():
    config = Config()

    assert config.model.hidden == 32
    assert config.training.epochs == 30


def test_load_from_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"hidden": 64}}))

    config = load_config(path)

    assert config.model.hidden == 64


def test_missing_values_use_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"hidden": 64}}))

    config = load_config(path)

    assert config.model.n_layers == 1
    assert config.training.lr == 1e-2


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"hiden": 64}}))  # typo voluta

    with pytest.raises(TypeError):
        load_config(path)


def test_unknown_section_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"modle": {}})


def test_save_and_load_roundtrip(tmp_path):
    config = Config.from_dict({"model": {"hidden": 16}, "training": {"epochs": 5}})
    path = tmp_path / "config.json"

    save_config(config, path)

    assert load_config(path) == config


def test_feature_columns_default_to_target():
    assert Config().data.feature_columns == ["value"]
    config = Config.from_dict({"data": {"target": "price"}})
    assert config.data.feature_columns == ["price"]


def test_feature_columns_roundtrip(tmp_path):
    config = Config.from_dict({"data": {"target": "a", "features": ["a", "b"]}})
    path = tmp_path / "config.json"

    save_config(config, path)

    assert load_config(path).data.feature_columns == ["a", "b"]
