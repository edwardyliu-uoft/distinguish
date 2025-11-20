"""Minimal training example using a tiny synthetic dataset.
Run with: python examples/minimal_train.py
"""

import os
import tempfile
from PIL import Image
from torch.utils.data import DataLoader

from distinguish.model import build_model
from distinguish.data import DistinguishDataset, default_transforms
from distinguish.train import Trainer, TrainerConfig
from distinguish.utils import get_device
from distinguish.io import save_model


def create_img(path: str, color=(128, 128, 128)):
    """Create a test image."""
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


def main():
    """Run a minimal training example with synthetic data."""
    device = get_device()
    model = build_model(pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "real")
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)
        for i in range(4):
            create_img(os.path.join(real_dir, f"real_{i}.jpg"), (100, 100, 100))
            create_img(os.path.join(fake_dir, f"fake_{i}.jpg"), (150, 150, 150))
        ds_train = DistinguishDataset(
            root=tmp, transform=default_transforms(train=True)
        )
        loader = DataLoader(ds_train, batch_size=2, shuffle=True)
        config = TrainerConfig(epochs=1, batch_size=2, lr=1e-3)
        trainer = Trainer(model, device=device, config=config)
        trainer.train(loader, loader)
        save_model(model, "minimal_weights.pt")
        print("Saved weights to minimal_weights.pt")


if __name__ == "__main__":
    main()
