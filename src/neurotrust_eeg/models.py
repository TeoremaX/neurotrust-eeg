from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyTCN(nn.Module):
    """Compact temporal convolutional network for EEG windows."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.blocks(z)
        z = self.pool(z).squeeze(-1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.head(z).squeeze(-1)


class PrototypeMemoryHead(nn.Module):
    """Prototype-based associative memory head.

    This is the first brain-inspired component of the project.
    Instead of using only a linear classifier, the model compares
    EEG embeddings against learned positive and negative prototypes.
    """

    def __init__(self, latent_dim: int, prototypes_per_class: int = 4):
        super().__init__()

        self.negative = nn.Parameter(torch.randn(prototypes_per_class, latent_dim) * 0.05)
        self.positive = nn.Parameter(torch.randn(prototypes_per_class, latent_dim) * 0.05)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        negative = F.normalize(self.negative, dim=-1)
        positive = F.normalize(self.positive, dim=-1)

        sim_negative = torch.max(z @ negative.T, dim=-1).values
        sim_positive = torch.max(z @ positive.T, dim=-1).values

        return self.scale * (sim_positive - sim_negative)


class TinyTCNWithPrototypeMemory(nn.Module):
    """TinyTCN encoder with prototype-based memory classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        prototypes_per_class: int = 4,
        num_blocks: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = TinyTCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.memory = PrototypeMemoryHead(
            latent_dim=hidden_channels,
            prototypes_per_class=prototypes_per_class,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.memory(z)