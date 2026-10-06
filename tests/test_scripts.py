import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pytorchlstm.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_train_then_predict(tmp_path):
    csv = tmp_path / "data.csv"
    t = np.arange(300)
    pd.DataFrame({"value": np.sin(t / 20)}).to_csv(csv, index=False)

    config = tmp_path / "config.json"
    config.write_text('{"model": {"hidden": 8}, "data": {"window": 10}, "training": {"epochs": 3}}')
    ckpt = tmp_path / "model.pt"

    train_out = run("train.py", "--config", str(config), "--csv", str(csv), "--output", str(ckpt))
    assert ckpt.exists()
    assert "Test MAE" in train_out.stdout

    predict_out = run("predict.py", "--checkpoint", str(ckpt), "--csv", str(csv))
    assert "Next value predicted" in predict_out.stdout

    plots = tmp_path / "plots"
    train_out = run(
        "train.py",
        "--config",
        str(config),
        "--csv",
        str(csv),
        "--output",
        str(ckpt),
        "--plots-dir",
        str(plots),
    )
    assert ckpt.exists()
    assert "Test MAE" in train_out.stdout
    for name in ("loss", "predictions", "scatter", "residuals"):
        assert (plots / f"{name}.png").exists()

    assert "Best epoch" in train_out.stdout


def test_multivariate_train_then_predict(tmp_path):
    csv = tmp_path / "data.csv"
    t = np.arange(300)
    pd.DataFrame({"y": np.sin(t / 20), "x1": np.cos(t / 20), "x2": np.sin((t + 5) / 20)}).to_csv(
        csv, index=False
    )
    config = tmp_path / "config.json"
    config.write_text('{"model": {"hidden": 8}, "data": {"window": 10}, "training": {"epochs": 3}}')
    ckpt = tmp_path / "model.pt"

    train_out = run(
        "train.py",
        "--config",
        str(config),
        "--csv",
        str(csv),
        "--output",
        str(ckpt),
        "--target",
        "y",
        "--features",
        "y",
        "x1",
        "x2",
        "--plots-dir",
        str(tmp_path / "plots"),
    )
    assert "['y', 'x1', 'x2']" in train_out.stdout

    loaded = load_checkpoint(ckpt)
    assert loaded.config.model.n_features == 3
    assert loaded.config.data.feature_columns == ["y", "x1", "x2"]

    predict_out = run("predict.py", "--checkpoint", str(ckpt), "--csv", str(csv))
    assert "Next value predicted for 'y'" in predict_out.stdout


def test_target_not_among_features(tmp_path):
    csv = tmp_path / "data.csv"
    t = np.arange(200)
    pd.DataFrame({"y": np.sin(t / 20), "x": np.sin((t + 1) / 20)}).to_csv(csv, index=False)
    config = tmp_path / "config.json"
    config.write_text('{"model": {"hidden": 8}, "data": {"window": 10}, "training": {"epochs": 2}}')
    ckpt = tmp_path / "model.pt"

    run(
        "train.py",
        "--config",
        str(config),
        "--csv",
        str(csv),
        "--output",
        str(ckpt),
        "--target",
        "y",
        "--features",
        "x",
    )
    predict_out = run("predict.py", "--checkpoint", str(ckpt), "--csv", str(csv))

    assert "Next value predicted for 'y'" in predict_out.stdout


def test_training_is_reproducible(tmp_path):
    csv = tmp_path / "data.csv"
    t = np.arange(200)
    pd.DataFrame({"value": np.sin(t / 20)}).to_csv(csv, index=False)
    config = tmp_path / "config.json"
    config.write_text('{"model": {"hidden": 8}, "data": {"window": 10}, "training": {"epochs": 3}}')

    outputs = [
        run(
            "train.py",
            "--config",
            str(config),
            "--csv",
            str(csv),
            "--output",
            str(tmp_path / f"model{i}.pt"),
        ).stdout.split("Model saved")[0]
        for i in range(2)
    ]

    assert outputs[0] == outputs[1]
