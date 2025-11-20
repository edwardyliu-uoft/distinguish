"""Smoke tests for training loop."""

import os
import tempfile
from PIL import Image
from torch.utils.data import DataLoader

from distinguish.data import DistinguishDataset, default_transforms
from distinguish.model import build_model
from distinguish.train import Trainer, TrainerConfig
from distinguish.utils import get_device


def create_img(path: str, color=(128, 128, 128)):
    """Create a test image."""
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


def test_smoke_training_loop():
    """Test that a minimal training loop executes without errors."""
    device = get_device()
    model = build_model(pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "real")
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)
        for i in range(2):
            create_img(os.path.join(real_dir, f"real_{i}.jpg"), (100, 100, 100))
            create_img(os.path.join(fake_dir, f"fake_{i}.jpg"), (150, 150, 150))
        ds = DistinguishDataset(root=tmp, transform=default_transforms(train=True))
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        config = TrainerConfig(epochs=1, batch_size=2, lr=1e-3)
        trainer = Trainer(model, device=device, config=config)
        result = trainer.train(loader, loader)
        assert "history" in result
        assert result["history"][0]["train_loss"] >= 0.0
