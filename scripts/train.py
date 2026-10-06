import argparse
from dataclasses import asdict, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytorchlstm.checkpoint import save_checkpoint
from pytorchlstm.config import Config, load_config
from pytorchlstm.data import column_stats, create_windows, select_columns
from pytorchlstm.evaluation import mae, rmse
from pytorchlstm.model import LSTMModel
from pytorchlstm.plotting import (
    plot_loss,
    plot_predictions,
    plot_residuals,
    plot_scatter,
    save_figure,
)
from pytorchlstm.training import fit, predict, set_seed


def denormalize(
    values: npt.NDArray[np.float32], mean: float, std: float
) -> npt.NDArray[np.float32]:
    return (values * std + mean).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to a JSON config")
    parser.add_argument("--csv", default="data/demo.csv")
    parser.add_argument(
        "--target",
        "--column",
        default=None,
        help="column to predict (overrides the config)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="input columns (overrides the config); defaults to the target only",
    )
    parser.add_argument("--output", default="models/lstm.pt")
    parser.add_argument("--plots-dir", default=None, help="if set, save evaluation plots here")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else Config()
    if args.target is not None:
        config = replace(config, data=replace(config.data, target=args.target))
    if args.features is not None:
        config = replace(config, data=replace(config.data, features=args.features))
    features = config.data.feature_columns

    # The input size of the model always follows the chosen feature columns
    config = replace(config, model=replace(config.model, n_features=len(features)))
    window = config.data.window

    df = pd.read_csv(args.csv)
    values = select_columns(df, features)
    target = select_columns(df, [config.data.target])[:, 0]

    n_train = int(len(values) * config.data.train_fraction)
    n_fit = int(n_train * (1 - config.data.val_fraction))

    # Statistics from the fitting part only
    mean, std = column_stats(values[:n_fit])
    t_mean, t_std = column_stats(target[:n_fit, np.newaxis])
    target_mean, target_std = float(t_mean[0]), float(t_std[0])
    values = ((values - mean) / std).astype(np.float32)
    target = ((target - target_mean) / target_std).astype(np.float32)

    X_fit, y_fit = create_windows(values[:n_fit], window, target[:n_fit])
    X_val, y_val = create_windows(
        values[n_fit - window : n_train], window, target[n_fit - window : n_train]
    )
    X_test, y_test = create_windows(values[n_train - window :], window, target[n_train - window :])

    print(f"Features: {features} -> target: {config.data.target!r}")
    set_seed(config.training.seed)  # for reproducibility
    model = LSTMModel(**asdict(config.model))
    result = fit(
        model,
        X_fit,
        y_fit,
        X_val,
        y_val,
        epochs=config.training.epochs,
        lr=config.training.lr,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        patience=config.training.patience,
    )
    status = "early stop" if result.stopped_early else "all epochs completed"
    print(f"Epochs executed: {len(result.train_loss)} ({status})")
    print(f"Best epoch: {result.best_epoch}")
    print(
        f"Min validation loss: {min(result.val_loss):.4f} | "
        f"final training loss: {result.train_loss[-1]:.4f}"
    )

    preds = predict(model, X_test)

    # Persistence baseline is the last observed target value before each step
    baseline = target[n_train - 1 : -1]
    print(f"Test MAE:  {mae(y_test, preds):.4f}")
    print(f"Test RMSE: {rmse(y_test, preds):.4f}")
    print(f"Baseline MAE:  {mae(y_test, baseline):.4f}")

    save_checkpoint(args.output, model, config, mean, std, target_mean, target_std)
    print(f"Model saved to {args.output}")

    if args.plots_dir:
        y_true_o = denormalize(y_test, target_mean, target_std)
        preds_o = denormalize(preds, target_mean, target_std)
        baseline_o = denormalize(baseline, target_mean, target_std)

        figures = {
            "loss.png": plot_loss(
                result.train_loss,
                result.val_loss,
                best_epoch=result.best_epoch,
                log_scale=True,
            ),
            "predictions.png": plot_predictions(y_true_o, preds_o, baseline_o),
            "scatter.png": plot_scatter(y_true_o, preds_o),
            "residuals.png": plot_residuals(y_true_o, preds_o),
        }
        for name, fig in figures.items():
            path = save_figure(fig, f"{args.plots_dir}/{name}")
            print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
