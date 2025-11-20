# distinguish

Binary classifier distinguishing real photographs from AI-generated images.

## Features

- PyTorch-based fine-tunable model (ResNet18 default, optional ResNet34 / EfficientNet-B0)
- Folder-structured dataset loader (`real/`, `fake/` or `ai_generated/`)
- Training with checkpointing, best-model saving, metrics CSV
- Evaluation (accuracy, precision, recall, F1)
- Classification CLI returning labels + probabilities
- Reproducible seeding & device auto-detection (CUDA/ROCm/MPS/CPU)

## Installation

```bash
pip install .
```

## Dataset Preparation (Kaggle 140k Real and Fake Faces)

Download and extract the dataset locally from Kaggle: <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>
Organize into a folder structure like:

```
faces_dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

If the Kaggle dataset extraction differs (e.g., single folder with mixed images), you can split randomly:

```python
import os, random, shutil
random.seed(42)
SRC = "raw_kaggle_folder"
DEST = "faces_dataset/train"
os.makedirs(os.path.join(DEST, "real"), exist_ok=True)
os.makedirs(os.path.join(DEST, "fake"), exist_ok=True)
# Example heuristic: filenames containing 'fake' -> fake, else real (adjust per dataset metadata)
for fname in os.listdir(SRC):
    src_path = os.path.join(SRC, fname)
    if not fname.lower().endswith(('.jpg','.png','.jpeg')): continue
    target = "fake" if "fake" in fname.lower() else "real"
    shutil.copy(src_path, os.path.join(DEST, target, fname))
```

Repeat similarly for validation/test splits.

## CLI Usage

After installation you have the `distinguish` command:

```bash
distinguish --version

distinguish --help

distinguish classify --weights weights.pt sample1.jpg sample2.jpg --out predictions.json
```

## API Usage

```python
from distinguish import build_model, DistinguishDataset, Trainer, Classifier, save_model, load_model
from torch.utils.data import DataLoader
from distinguish.train import TrainerConfig

model = build_model("resnet18", pretrained=True)
train_ds = DistinguishDataset("faces_dataset/train")
val_ds = DistinguishDataset("faces_dataset/val", transform=None)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
config = TrainerConfig(epochs=2, lr=1e-4)
trainer = Trainer(model, config=config)
trainer.train(train_loader, val_loader)
save_model(model, "weights.pt")

classifier = Classifier("weights.pt")
results = classifier.predict(["sample1.jpg", "sample2.jpg"])
print(results)
```

## Minimal Quick Test

Run a tiny synthetic training session:

```bash
python examples/minimal_train.py
python examples/classify_example.py minimal_weights.pt examples/minimal_train.py  # (nonsense example path)
```

## Device Detection

Automatically selects CUDA/ROCm (via `torch.cuda.is_available()`), MPS (Apple Silicon), else CPU. Logs chosen device.

## Metrics Logging

Metrics are saved to `runs/metrics.csv` after training. Best weights at `runs/best_model.pt`.

## Testing

```bash
pytest -q
```

Includes tests for dataset, model IO, and smoke training.

## Contributing

PRs welcome for new backbones, better augmentation, or more metrics.

## License

MIT

## Changelog

See `CHANGELOG.md`.
