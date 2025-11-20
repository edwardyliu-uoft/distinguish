"""Data loading utilities for the distinguish package.

Defines ``DistinguishDataset`` for folder-structured face image datasets.
Expected directory layout:
    root/
        real/
            img1.jpg
            ...
        fake/  (alias: ai_generated/)
            img2.jpg
            ...

The dataset returns (image_tensor, label_int) where label_int is 0 for real
and 1 for AI-generated (fake).
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"real": 0, "fake": 1, "ai_generated": 1}

DEFAULT_IMAGE_SIZE = 224


def default_transforms(train: bool = True) -> transforms.Compose:
    """Return default torchvision transforms for faces.

    Parameters
    ----------
    train: bool
        If True include data augmentation.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DistinguishDataset(Dataset):
    """Folder-based dataset for real vs AI-generated face images.

    Parameters
    ----------
    root: str
        Root directory containing subfolders for each class.
    transform: optional callable
        Transform applied to PIL images.
    extensions: list[str]
        File extensions to consider.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Optional[List[str]] = None,
    ) -> None:
        self.root = root
        self.transform = transform or default_transforms(train=True)
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp"]
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.samples: List[Tuple[str, int]] = []
        for class_name, label in LABEL_MAP.items():
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                ext_match = any(fname.lower().endswith(ext) for ext in self.extensions)
                if os.path.isfile(fpath) and ext_match:
                    self.samples.append((fpath, label))
        if not self.samples:
            raise RuntimeError(f"No image files found under {root}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
