# pytorchlstm

PyTorch implementation of a Long Short-Term Memory (LSTM) model for one-step-ahead time-series forecasting.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Using the library from Python](#using-the-library-from-python)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations](#limitations)

## Features

- Stacked LSTM regressor with configurable hidden size and number of layers.
- Multivariate input: any set of CSV columns as features, any column as the
  target.
- Train/validation/test split with no shuffling and no data leakage:
  normalization statistics come from the fitting portion only.
- Early stopping on validation loss, with the best weights restored.
- Configurable seed for reproducible runs.
- Evaluation with MAE and RMSE.
- Single-file checkpoints containing weights, config (including column names) and
  normalization stats.
- Optional evaluation plots: loss curves, predictions, predicted-vs-actual
  scatter, residuals.
- JSON configuration with defaults for every hyperparameter.

## Requirements

- Python >= 3.12
- `torch`, `numpy`, `pandas`, `matplotlib`

Training runs on the CPU.

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate

# Runtime only
pip install -e .

# Runtime and development tools (pytest, ruff, mypy, pandas-stubs)
pip install -e ".[dev]"
```

## Quick start

Run all commands from the project root.

```bash
# 1. Generate a demo dataset and save it to `data/demo.csv`
python scripts/make_demo_data.py

# 2. Run a training session with specified hyperparameters configuration and plots directory
python scripts/train.py --config configs/default.json --plots-dir plots

# 3. Predict the value that follows the end of the series
python scripts/predict.py
```

Example output of the multivariate run:

```
Features: ['value', 'lead'] -> target: 'value'
Epochs executed: 11 (early stop)
Best epoch: 6
Min validation loss: 0.0241 | final training loss: 0.0273
Test MAE:  0.1078
Test RMSE: 0.1354
Baseline MAE:  0.1387
Model saved to models/lstm.pt
Plot saved to plots/loss.png
Plot saved to plots/predictions.png
Plot saved to plots/scatter.png
Plot saved to plots/residuals.png
```

and prediction:

```
Next value predicted for 'value': -0.3396
```

On the demo data the extra `lead` column helps: the univariate run reaches a test MAE of 0.118,
the multivariate one 0.108.

> [!NOTE]
> The losses and the MAE/RMSE that `train.py` prints are in **normalized**
> units (z-scores computed on the training data). The plots and
> `predict.py` output are in the **original** units.

## Command-line usage

### `scripts/train.py`

Trains a model, evaluates it on the test split, saves a checkpoint and
optionally saves plots.

| Option          | Default          | Description                                         |
|-----------------|------------------|-----------------------------------------------------|
| `--config`      | *(none)*         | Path to a JSON config. Without it, the built-in defaults are used. |
| `--csv`         | `data/demo.csv`  | Input CSV file.                                     |
| `--target`      | *(from config)*  | Column to predict. `--column` is an alias. Overrides `data.target`. |
| `--features`    | *(from config)*  | One or more input columns, space-separated. Overrides `data.features`. |
| `--output`      | `models/lstm.pt` | Where to write the checkpoint. Parent dirs are created. |
| `--plots-dir`   | *(none)*         | If given, evaluation plots are saved in this directory. |

Usage examples:
```bash
# Univariate: predict `price` from past prices
python scripts/train.py --csv my_data.csv --target price \
    --config configs/default.json --output models/price.pt --plots-dir plots/price

# Example: predict `price` from past price, volume and temperature
python scripts/train.py --csv my_data.csv --target price --features price volume temperature \
    --config configs/default.json --output models/price_multi.pt
```

The model's input size (`model.n_features`) is automatically set to the number of feature columns.

### `scripts/predict.py`

Loads a checkpoint, takes the last `window` rows of the feature columns, and predicts the next
value of the target.

| Option          | Default          | Description                          |
|-----------------|------------------|--------------------------------------|
| `--checkpoint`  | `models/lstm.pt` | Checkpoint produced by `train.py`.   |
| `--csv`         | `data/demo.csv`  | CSV with the series to extend.       |

The feature and target column names, the window length, and the normalization all come from the
checkpoint. The CSV file must contain the feature columns and have at least `window` rows.

### `scripts/make_demo_data.py`

Writes `data/demo.csv` with two columns for `t = 0 ... 999` (seed 0):

- `value`: `sin(t / 20) + 0.1 · noise`, the series to forecast
- `lead`: `sin((t + 5) / 20) + 0.1 · noise`, the same signal 5 steps ahead, i.e. a leading
  indicator that a multivariate model can exploit

### Input format

Any CSV readable by `pandas.read_csv` with numeric columns works:

```csv
value,lead
0.0126,0.3658
0.0368,0.3259
0.1639,0.3970
...
```

Rows are assumed to be in time order and evenly spaced. Columns that are neither features nor
the target are ignored. A missing column raises an error listing the available ones.

## Configuration

Hyperparameters configs are JSON files with up to three sections: `model`, `data` and
`training`. Every key is optional and unknown sections or keys raise an error.

Univariate configuration example:

```json
{
  "model": {"hidden": 64, "n_layers": 2},
  "data": {"window": 30, "train_fraction": 0.8},
  "training": {"epochs": 40, "lr": 0.005, "batch_size": 32, "seed": 0}
}
```

Multivariate configuration example (see `configs/default.json`):

```json
{
  "model": {"hidden": 64, "n_layers": 2},
  "data": {
    "window": 30,
    "train_fraction": 0.8,
    "target": "value",
    "features": ["value", "lead"]
  },
  "training": {"epochs": 40, "lr": 0.005, "batch_size": 32, "seed": 0}
}
```

### All options

| Section    | Key              | Default | Description |
|------------|------------------|---------|-------------|
| `model`    | `n_features`     | `1`     | Input features per time step. `train.py` sets it to the number of feature columns. |
| `model`    | `hidden`         | `32`    | LSTM hidden size. |
| `model`    | `n_layers`       | `1`     | Number of stacked LSTM layers. |
| `data`     | `window`         | `30`    | Number of past values used to predict the next one. |
| `data`     | `train_fraction` | `0.8`   | Share of the series used for training + validation; the rest is the test set. |
| `data`     | `val_fraction`   | `0.15`  | Share of the *training part* held out for validation/early stopping. |
| `data`     | `target`         | `"value"` | Column to predict. |
| `data`     | `features`       | `null`  | List of input columns. `null` means just the target (univariate). |
| `training` | `epochs`         | `30`    | Maximum number of epochs. |
| `training` | `lr`             | `0.01`  | Adam learning rate. |
| `training` | `batch_size`     | `32`    | Mini-batch size. |
| `training` | `seed`           | `0`     | Random seed for PyTorch and NumPy. |
| `training` | `patience`       | `5`     | Epochs without validation improvement before stopping. Must be ≥ 1. |

The defaults in this table are used when no `--config` is
passed.

## How it works

### Data split

The series is split in time order and never shuffled across splits:

```
|<──────────── train_fraction ────────────>|<──── test ────>|
|<──── fit (1 - val_fraction) ────>|< val >|
```

### Preprocessing

1. **Normalization.** Mean and standard deviation are computed per column on the *fit*
   portion only and applied to the whole data. The target gets its own statistics, used to convert predictions back to original units. A constant column gets a std of 1 to avoid dividing by zero.
2. **Windowing.** `create_windows()` turns the feature matrix into pairs
   `X[i] = features[i : i + window]` (shape `(window, n_features)`) and
   `y[i] = target[i + window]`. Validation and test windows borrow the last `window` rows of the
   preceding split as context.

### Model

```
input (batch, window, n_features) ─> LSTM (n_layers, hidden) ─> last time step ─> Linear(hidden, 1) ─> (batch,)
```

### Training

- Loss: **MSE**, mean squared error. Optimizer: Adam.
- Each epoch trains on shuffled mini-batches of the fit windows, then computes
  the validation loss.
- If the validation loss hasn't improved for `patience` epochs, training stops.
  The weights from the best epoch are restored either way.

### Evaluation

The test set is scored with:

- **MAE**, mean absolute error
- **RMSE**, root mean squared error
- **Baseline MAE**, the MAE of always predicting the last observed target value (persistence
  forecast). A useful model should beat it.

### Checkpoints

`save_checkpoint` writes one `.pt` file containing:

| Key          | Content                                 |
|--------------|-----------------------------------------|
| `state_dict` | Model weights                           |
| `config`     | The full `Config` as a dict, including feature and target column names |
| `mean`, `std`| Per-feature normalization statistics from training (lists) |
| `target_mean`, `target_std` | Normalization statistics of the target |

`load_checkpoint` rebuilds the model from the stored config, loads the weights
(with `weights_only=True`), puts it in eval mode and returns a `Checkpoint`
dataclass. Checkpoints saved before multivariate support (scalar `mean`/`std`, no target
statistics) still load.

### Plots

With `--plots-dir`, `train.py` saves four PNGs (150 dpi) about the target, all in original
units:

| File              | Shows |
|-------------------|-------|
| `loss.png`        | Train and validation loss per epoch (log scale), with the best epoch marked |
| `predictions.png` | Actual test values vs LSTM predictions vs the persistence baseline |
| `scatter.png`     | Predicted vs actual; a perfect model lies on the diagonal |
| `residuals.png`   | Residuals over time and their histogram |

## Script usage

The modules in `src/pytorchlstm` can be used directly:

```python
import pandas as pd
from dataclasses import replace
from pytorchlstm.data import column_stats, select_columns

df = pd.read_csv("data/demo.csv")
features = select_columns(df, ["value", "lead"])  
target = select_columns(df, ["value"])[:, 0]  

mean, std = column_stats(features[:700]) 
features = (features - mean) / std
target = (target - mean[0]) / std[0]

X, y = create_windows(features, 30, target)  

config = Config()
config = replace(
    config,
    model=replace(config.model, n_features=2),
    data=replace(config.data, target="value", features=["value", "lead"]),
)

set_seed(config.training.seed)
model = LSTMModel(**asdict(config.model))
result = fit(model, X_train, y_train, X_val, y_val, epochs=50, patience=5)
print(result.best_epoch, result.stopped_early)

preds = predict(model, X_test)
print(f"MAE {mae(y_test, preds):.4f}  RMSE {rmse(y_test, preds):.4f}")

save_figure(plot_predictions(y_test, preds), "plots/example.png")
save_checkpoint("models/multi.pt", model, config, mean, std)
```

### API overview

| Module       | Contents |
|--------------|----------|
| `config`     | `ModelConfig`, `DataConfig` (with the `feature_columns` property), `TrainingConfig`, `Config` (with `Config.from_dict`), `load_config(path)`, `save_config(config, path)` |
| `data`       | `create_windows(series, length, target=None) -> (X, y)`, `column_stats(values) -> (mean, std)`, `select_columns(df, columns)` |
| `model`      | `LSTMModel(n_features=1, hidden=32, n_layers=1)` |
| `training`   | `fit(...) -> FitResult` (early stopping), `train(...) -> list[float]` (fixed epochs, no validation), `predict(model, X)`, `set_seed(seed)` |
| `evaluation` | `mae(y_true, y_pred)`, `rmse(y_true, y_pred)` |
| `checkpoint` | `save_checkpoint(path, model, config, mean, std, target_mean=None, target_std=None)`, `load_checkpoint(path) -> Checkpoint` |
| `plotting`   | `plot_loss`, `plot_predictions`, `plot_scatter`, `plot_residuals` (each returns a matplotlib `Figure`), `save_figure(fig, path, dpi=150)` |

`FitResult` has `train_loss`, `val_loss`, `best_epoch` (1-based) and
`stopped_early`. The plotting functions build `Figure` objects directly
instead of using `pyplot`.

## Project layout

```
pytorchlstm/
├── configs/
│   └── default.json         # example configuration
├── data/                    # input CSVs (contents git-ignored)
├── models/                  # saved checkpoints
├── plots/                   # generated plots (contents git-ignored)
├── scripts/
│   ├── make_demo_data.py    # generate data/demo.csv
│   ├── train.py             # train, evaluate, save checkpoint and plots
│   └── predict.py           # predict the next value from a checkpoint
├── src/pytorchlstm/
│   ├── checkpoint.py        # save/load model + config + normalization
│   ├── config.py            # dataclass config and JSON I/O
│   ├── data.py              # column selection, normalization stats, sliding windows
│   ├── evaluation.py        # MAE, RMSE
│   ├── model.py             # LSTMModel
│   ├── plotting.py          # evaluation figures
│   └── training.py          # fit (early stopping), train, predict
├── tests/                   # pytest suite
├── pyproject.toml
├── .gitignore
└── README.md
```

## Development

Install the dev extras first (`pip install -e ".[dev]"`), then:

```bash
pytest              # run the test suite
ruff check .        # lint (pycodestyle, pyflakes, isort, bugbear)
ruff format .       # format, line length 100
mypy src scripts    # type-check
```

The tests cover config parsing, windowing, the model, training and early
stopping, metrics, checkpoint round-trips, plotting, and an end-to-end
`train.py` → `predict.py` runs (univariate, multivariate, and target not among the features)
on small synthetic series.

## Limitations

- **CPU only.** Tensors are never moved to a GPU. That's fine at this scale, but
  big datasets would need device handling added.
- **Bitwise reproducibility is per machine.** A different number of CPU threads or a different
  PyTorch build can change results in the last few digits (about 1e-8), because floating-point
  sums run in a different order.
- **Training metrics are normalized.** `train.py` prints MAE/RMSE in z-score
  units. Multiply by the training `std` to get original units.
