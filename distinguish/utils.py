"""Utility helpers for distinguish."""

from __future__ import annotations
import os
import csv
import random
import logging
from typing import Iterable, Dict, Any
import numpy as np
import torch

logger = logging.getLogger("distinguish")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_device() -> torch.device:
    """Return best available computation device.

    Prefers CUDA if available. For AMD ROCm builds, ``torch.cuda.is_available``
    should also return True. Falls back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU device: %s", torch.cuda.get_device_name(0))
        return device
    # Apple MPS (not requested but harmless)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")


def write_metrics_csv(
    path: str, rows: Iterable[Dict[str, Any]], fieldnames: Iterable[str]
) -> None:
    """Write metrics rows to CSV."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_binary_metrics(preds, targets) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 for binary predictions.

    Parameters
    ----------
    preds: torch.Tensor
        Raw logits or probabilities.
    targets: torch.Tensor
        Ground truth labels (0/1).
    """
    if preds.ndim != 1:
        preds = preds.view(-1)
    probs = torch.sigmoid(preds)
    labels = (probs >= 0.5).float()
    targets = targets.float()
    tp = ((labels == 1) & (targets == 1)).sum().item()
    tn = ((labels == 0) & (targets == 0)).sum().item()
    fp = ((labels == 1) & (targets == 0)).sum().item()
    fn = ((labels == 0) & (targets == 1)).sum().item()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
