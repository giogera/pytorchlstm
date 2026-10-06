"""LSTM model."""

import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch, time, hidden)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,)
