"""PyTorch model used by both training and single-molecule inference."""

from __future__ import annotations

import torch
from torch import nn


class ESOLMLP(nn.Module):
    """1024 -> 256 -> 64 -> 1 regression network for ESOL logS."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
