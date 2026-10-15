from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


class Expert(nn.Module, ABC):
    """Interface: a causal sequence model mapping hidden states [B, T, D] to [B, T, D]."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden states without looking at future positions."""


@dataclass
class Routing:
    """Router result: per-position expert weights [B, T, E] and named, already-weighted auxiliary losses."""
    weights: torch.Tensor
    losses: dict = field(default_factory=dict)


class Router(nn.Module, ABC):
    """Interface: mixing weights over experts for every position, using only that position's prefix."""

    @abstractmethod
    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> Routing:
        """Route embeddings x [B, T, D] (with their token ids [B, T]) to experts."""
