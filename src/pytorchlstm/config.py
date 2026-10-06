"""Configuration dataclasses for model, data, training, and hyperparameters."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    n_features: int = 1
    hidden: int = 32
    n_layers: int = 1


@dataclass
class DataConfig:
    window: int = 30
    train_fraction: float = 0.8
    val_fraction: float = 0.15         # share of the train part used for validation
    target: str = "value"              # column to predict
    features: list[str] | None = None  # input columns; None means [target]

    @property
    def feature_columns(self) -> list[str]:
        """Input columns, in the order the model sees them."""
        return list(self.features) if self.features else [self.target]


@dataclass
class TrainingConfig:
    epochs: int = 30
    lr: float = 1e-2
    batch_size: int = 32
    seed: int = 0
    patience: int = 5  # epochs without improvement before stopping


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        unknown = set(raw) - {"model", "data", "training"}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )


def load_config(path: str | Path) -> Config:
    """Load a config from a JSON file. Missing values use the defaults."""
    raw = json.loads(Path(path).read_text())
    return Config.from_dict(raw)


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2))
