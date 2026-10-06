"""Predict the next value in a time series using a trained LSTM model."""

import argparse

import numpy as np
import pandas as pd

from pytorchlstm.checkpoint import load_checkpoint
from pytorchlstm.data import select_columns
from pytorchlstm.training import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/lstm.pt")
    parser.add_argument("--csv", default="data/demo.csv")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    window = ckpt.config.data.window
    features = ckpt.config.data.feature_columns

    # Column names and order come from the checkpoint
    values = select_columns(pd.read_csv(args.csv), features)
    if len(values) < window:
        raise SystemExit(f"Need at least {window} rows, got {len(values)}")
    mean = np.array(ckpt.mean, dtype=np.float32)
    std = np.array(ckpt.std, dtype=np.float32)
    last = ((values[-window:] - mean) / std).astype(np.float32)
    X = last[np.newaxis]  # (1, window, n_features)

    prediction = float(predict(ckpt.model, X)[0]) * ckpt.target_std + ckpt.target_mean
    print(f"Next value predicted for {ckpt.config.data.target!r}: {prediction:.4f}")


if __name__ == "__main__":
    main()
