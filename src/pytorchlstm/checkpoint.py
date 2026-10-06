"""Save and load model checkpoints."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pytorchlstm.config import Config
from pytorchlstm.model import LSTMModel


@dataclass
class Checkpoint:
    model: LSTMModel
    config: Config
    mean: list[float]  # one value per feature column
    std: list[float]
    target_mean: float
    target_std: float


def _as_list(values: float | Sequence[float] | np.ndarray) -> list[float]:
    return [float(v) for v in np.atleast_1d(np.asarray(values))]


def _target_stats(config: Config, mean: list[float], std: list[float]) -> tuple[float, float]:
    """Target statistics taken from the feature statistics, when possible."""
    features = config.data.feature_columns
    if config.data.target not in features:
        raise ValueError(
            "target_mean and target_std are required when the target is not one of the features"
        )
    i = features.index(config.data.target)
    return mean[i], std[i]


def save_checkpoint(
    path: str | Path,
    model: LSTMModel,
    config: Config,
    mean: float | Sequence[float] | np.ndarray,
    std: float | Sequence[float] | np.ndarray,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> None:
    """Save weights, config and normalization statistics in a single file.

    `mean` and `std` hold one value per feature column. The target statistics
    can be omitted when the target is one of the features.
    """
    mean_list, std_list = _as_list(mean), _as_list(std)
    if target_mean is None or target_std is None:
        target_mean, target_std = _target_stats(config, mean_list, std_list)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "mean": mean_list,
            "std": std_list,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Rebuild the model from a checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    config = Config.from_dict(raw["config"])
    model = LSTMModel(**asdict(config.model))
    model.load_state_dict(raw["state_dict"])
    model.eval()

    # Older univariate checkpoints store scalar stats and no target stats
    mean, std = _as_list(raw["mean"]), _as_list(raw["std"])
    if "target_mean" in raw:
        target_mean, target_std = float(raw["target_mean"]), float(raw["target_std"])
    else:
        target_mean, target_std = _target_stats(config, mean, std)
    return Checkpoint(model, config, mean, std, target_mean, target_std)
