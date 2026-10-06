"""Training utilities."""

import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(
    model: nn.Module,
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Train the model and return the mean loss of each epoch."""
    set_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(dataset))
    return history


def predict(model: nn.Module, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return the model predictions as a NumPy array."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X))
    return out.numpy()


@dataclass
class FitResult:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int  # 1-based
    stopped_early: bool


def _mean_loss(
    model: nn.Module,
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    loss_fn: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model(torch.from_numpy(X)), torch.from_numpy(y))
    return float(loss.item())


def fit(
    model: nn.Module,
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    X_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.float32],
    epochs: int = 100,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 0,
    patience: int = 5,
) -> FitResult:
    """Train with early stopping and restore the best weights.

    Training stops when the validation loss has not improved for
    `patience` consecutive epochs.
    """
    if patience < 1:
        raise ValueError("patience must be at least 1")

    set_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loss: list[float] = []
    val_loss: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss.append(total / len(dataset))

        current = _mean_loss(model, X_val, y_val, loss_fn)
        val_loss.append(current)

        if current < best_val:
            best_val = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    model.eval()
    return FitResult(train_loss, val_loss, best_epoch, stopped_early)
