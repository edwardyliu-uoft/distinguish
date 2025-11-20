"""Prediction utilities and Classifier class."""

from __future__ import annotations
import os
import json
import logging
from typing import List, Dict, Any
from PIL import Image
import torch
from torchvision import transforms

from .model import build_model
from .io import load_model
from .utils import get_device

logger = logging.getLogger("distinguish")

PRED_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

LABEL_NAMES = {0: "real", 1: "ai_generated"}


class Classifier:
    """Convenience wrapper for loading a trained model and performing inference.

    Parameters
    ----------
    weights_path: str
        Path to a saved model (state dict) file.
    device: optional torch.device
        Device to run inference on. Auto-selected if not provided.
    backbone: str
        Backbone architecture used during training.
    """

    def __init__(
        self,
        weights_path: str,
        device: torch.device | None = None,
        backbone: str = "resnet18",
    ) -> None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        self.device = device or get_device()
        self.model = build_model(backbone=backbone, pretrained=False)
        load_model(self.model, weights_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: List[str]) -> List[Dict[str, Any]]:
        """Predict labels for a list of image file paths."""
        results = []
        with torch.no_grad():
            for path in images:
                if not os.path.isfile(path):
                    logger.warning("Skipping missing file: %s", path)
                    continue
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    tensor = PRED_TRANSFORM(img).unsqueeze(0).to(self.device)
                    logit = self.model(tensor).view(-1)[0]
                    prob = torch.sigmoid(logit).item()
                    label = 1 if prob >= 0.5 else 0
                    results.append(
                        {
                            "path": path,
                            "label": LABEL_NAMES[label],
                            "score": prob,
                        }
                    )
        return results


def save_predictions_json(results: List[Dict[str, Any]], out_path: str) -> None:
    """Save prediction results to JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
