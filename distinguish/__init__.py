"""distinguish package

Binary classifier to distinguish between real photographs versus AI-generated (synthetic) images.

Public API:
    build_model, DistinguishDataset, Trainer, Classifier,
    save_model, load_model
"""

from .model import build_model
from .data import DistinguishDataset
from .train import Trainer
from .predict import Classifier
from .io import save_model, load_model

__all__ = [
    "build_model",
    "DistinguishDataset",
    "Trainer",
    "Classifier",
    "save_model",
    "load_model",
]
