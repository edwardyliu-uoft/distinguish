"""Tests for data loading utilities."""

import os
import tempfile
from PIL import Image
import torch
from distinguish.data import DistinguishDataset, default_transforms


def create_img(path: str, color=(128, 128, 128)):
    """Create a test image."""
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


def test_dataset_loading():
    """Test that DistinguishDataset loads images correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "real")
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)
        for i in range(3):
            create_img(os.path.join(real_dir, f"real_{i}.jpg"), (100, 100, 100))
            create_img(os.path.join(fake_dir, f"fake_{i}.jpg"), (150, 150, 150))
        ds = DistinguishDataset(root=tmp, transform=default_transforms(train=False))
        assert len(ds) == 6
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert y.item() in (0.0, 1.0)
