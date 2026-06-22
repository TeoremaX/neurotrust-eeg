from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)