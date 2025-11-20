"""Model persistence utilities."""

from __future__ import annotations
import os
import torch
from torch import nn


def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict to path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """Load model state dict from path into provided model instance."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    return model
