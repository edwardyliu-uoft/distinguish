# distinguish

**A PyTorch-based binary classifier for distinguishing real photographs from AI-generated synthetic images.**

Built for detecting deepfakes and AI-generated images using transfer learning with pretrained CNN backbones. Includes a complete training pipeline, evaluation metrics, command-line interface, and interactive Jupyter notebook demonstration.

---

## 🚀 Features

- **Multiple CNN Backbones**: ResNet18 (default), ResNet34, EfficientNet-B0 with ImageNet pretraining
- **Simple Dataset Loader**: Folder-structured dataset loader supporting `real/` and `fake/` (or `ai_generated/`) subdirectories
- **Complete Training Pipeline**: Automatic checkpointing, best-model tracking, comprehensive metrics logging (accuracy, precision, recall, F1)
- **Flexible Training Configuration**: Configurable optimizers (Adam/SGD), learning rates, batch sizes, gradient clipping, mixed precision
- **Command-Line Interface**: Easy-to-use CLI for inference
- **Python API**: Clean programmatic interface for integration into larger projects
- **Device Auto-detection**: Automatic selection of CUDA/ROCm GPU, Apple MPS (M1/M2), or CPU
- **Reproducible Results**: Built-in seeding for reproducible experiments
- **Interactive Demo**: Comprehensive Jupyter notebook with visualization and analysis

---

## 📦 Installation

### Basic Installation

```bash
pip install .
```

### Development Installation

For development with testing tools:

```bash
pip install -e ".[dev]"
```

### Jupyter Notebook Support

For running the interactive demo notebook:

```bash
pip install ".[ipynb]"
```

---

## 📊 Dataset Preparation

The dataset should be organized in a folder structure with separate `real/` and `fake/` subdirectories:

```text
datasets/faces/
  train/
    real/
      img001.jpg
      img002.jpg
      ...
    fake/
      gen001.jpg
      gen002.jpg
      ...
  valid/
    real/
    fake/
  test/
    real/
    fake/
```

### Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.bmp`

### Example Dataset Sources

- [Real vs AI Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset)
- Custom datasets following the same folder structure

### Dataset Splitting Script

If you have a single folder with mixed images:

```python
import os
import random
import shutil

random.seed(42)

def split_dataset(src_folder, dest_root, train_ratio=0.7, val_ratio=0.15):
    """Split mixed dataset into train/val/test with real/fake subdirectories."""
    # Organize files by label
    files = os.listdir(src_folder)
    real_files = [f for f in files if 'real' in f.lower()]
    fake_files = [f for f in files if 'fake' in f.lower()]
    
    for label, file_list in [('real', real_files), ('fake', fake_files)]:
        random.shuffle(file_list)
        n = len(file_list)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        
        splits = {
            'train': file_list[:train_n],
            'valid': file_list[train_n:train_n + val_n],
            'test': file_list[train_n + val_n:]
        }
        
        for split, files in splits.items():
            split_dir = os.path.join(dest_root, split, label)
            os.makedirs(split_dir, exist_ok=True)
            for fname in files:
                src = os.path.join(src_folder, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copy(src, dst)

split_dataset("raw_kaggle_folder", "datasets/faces")
```

---

## 🖥️ Command-Line Interface

After installation, the `distinguish` command is available:

### Check Version

```bash
distinguish --version
```

### Get Help

```bash
distinguish --help
```

### Classify Images

```bash
distinguish classify \
  --weights models/resnet18.pt \
  --backbone resnet18 \
  --out predictions.json \
  image1.jpg image2.jpg image3.png
```

**Output Example:**

```json
[
  {
    "path": "image1.jpg",
    "label": "real",
    "score": 0.9234
  },
  {
    "path": "image2.jpg",
    "label": "ai_generated",
    "score": 0.1456
  }
]
```

---

## 🐍 Python API Usage

### Quick Start: Training a Model

```python
from distinguish import build_model, DistinguishDataset, Trainer
from distinguish.data import default_transforms
from distinguish.train import TrainerConfig
from distinguish.io import save_model
from torch.utils.data import DataLoader

# Build model with pretrained backbone
model = build_model(backbone="resnet18", pretrained=True)

# Load datasets
train_ds = DistinguishDataset(
    "datasets/faces/train",
    transform=default_transforms(train=True)
)
valid_ds = DistinguishDataset(
    "datasets/faces/valid",
    transform=default_transforms(train=False)
)

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=4)

# Configure training
config = TrainerConfig(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    weight_decay=1e-4,
    optimizer="adam",
    out_dir="runs/experiment"
)

# Train model
trainer = Trainer(model, config=config)
result = trainer.train(train_loader, valid_loader)

# Save trained model
save_model(model, "my_model.pt")
print(f"Best validation accuracy: {result['best_metric']:.4f}")
```

### Running Inference

```python
from distinguish import Classifier

# Load trained model
classifier = Classifier("my_model.pt", backbone="resnet18")

# Predict on images
results = classifier.predict([
    "test_image1.jpg",
    "test_image2.jpg",
    "test_image3.png"
])

# Display results
for r in results:
    label = r['label']
    confidence = r['score']
    print(f"{r['path']}: {label} (confidence: {confidence:.2%})")
```

### Advanced: Custom Training Loop

```python
from distinguish import build_model, load_model
from distinguish.utils import compute_binary_metrics, get_device
import torch

device = get_device()
model = build_model("efficientnet_b0", pretrained=True).to(device)

# ... your custom training code ...

# Evaluate
model.eval()
with torch.no_grad():
    preds = []
    targets = []
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images).view(-1)
        preds.append(logits.cpu())
        targets.append(labels.cpu())

logits = torch.cat(preds)
labels = torch.cat(targets)
metrics = compute_binary_metrics(logits, labels)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1 Score: {metrics['f1']:.4f}")
```

---

## 📓 Interactive Demo Notebook

A comprehensive Jupyter notebook (`demo.ipynb`) demonstrates the complete workflow:

1. **Dataset Exploration**: Visualize class distribution and sample images
2. **Model Building**: Load pretrained ResNet18 backbone
3. **Training**: Train with progress tracking and metric logging
4. **Visualization**: Plot training curves (loss, accuracy, precision, recall, F1)
5. **Evaluation**: Test set evaluation with confusion matrix
6. **Predictions**: Sample prediction visualization
7. **Performance Summary**: Comprehensive metric comparison across splits

**Run the demo:**

```bash
# Install notebook dependencies
pip install -e ".[ipynb]"

# Launch Jupyter
jupyter notebook demo.ipynb

# Or use VS Code with Jupyter extension
code demo.ipynb
```

**Fast Mode** for quick experiments (uses subset of data):

```bash
export DISTINGUISH_FAST=1  # Unix/Mac
set DISTINGUISH_FAST=1     # Windows
```

---

## 🧪 Testing

Run the complete test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_data.py
```

### Test Coverage

- **`test_data.py`**: Dataset loading and transforms
- **`test_model_io.py`**: Model saving/loading
- **`test_train_smoke.py`**: End-to-end training smoke test

---

## 📝 Examples

### Minimal Training Script

Quick synthetic example for testing:

```bash
python examples/minimal_train.py
```

This creates a tiny synthetic dataset and trains for 1 epoch, saving weights to `minimal_weights.pt`.

### Classification Script

Classify images using trained weights:

```bash
python examples/classify_example.py runs/demo/best_model.pt \
    datasets/faces/test/real/00001.jpg \
    datasets/faces/test/fake/0AEIDNSBKD.jpg
```

---

## ⚙️ Configuration Reference

### TrainerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | `1e-4` | Learning rate |
| `batch_size` | int | `32` | Training batch size |
| `epochs` | int | `5` | Number of training epochs |
| `weight_decay` | float | `0.0` | L2 regularization weight |
| `optimizer` | str | `"adam"` | Optimizer type (`"adam"` or `"sgd"`) |
| `mixed_precision` | bool | `False` | Enable mixed precision training (CUDA only) |
| `out_dir` | str | `"runs"` | Output directory for checkpoints and metrics |
| `checkpoint_path` | str | `None` | Path to checkpoint for resuming |
| `resume` | bool | `False` | Resume training from checkpoint |
| `grad_clip` | float | `None` | Gradient clipping threshold |
| `seed` | int | `42` | Random seed for reproducibility |

### Supported Backbones

- **`resnet18`** (default): 11.7M parameters, fast training
- **`resnet34`**: 21.8M parameters, better accuracy
- **`efficientnet_b0`**: 5.3M parameters, efficient and accurate

---

## 🔧 Device Detection

The package automatically detects and uses the best available compute device:

1. **NVIDIA CUDA**: Detected via `torch.cuda.is_available()`
2. **AMD ROCm**: Also detected via CUDA interface (PyTorch ROCm builds)
3. **Apple MPS**: M1/M2 GPU acceleration on macOS
4. **CPU**: Fallback for all systems

Device selection is logged at runtime:

```text
[INFO] Using GPU device: NVIDIA GeForce RTX 3080
[INFO] Using Apple MPS device
[INFO] Using CPU device
```

---

## 📈 Metrics & Logging

During training, metrics are logged to the console and saved to CSV:

### Console Output

```text
Epoch 1: train_loss=0.3245 train_acc=0.8532 val_acc=0.8901
Epoch 2: train_loss=0.2134 train_acc=0.9123 val_acc=0.9245
...
```

### CSV Metrics (`runs/experiment/metrics.csv`)

| epoch | train_loss | train_accuracy | train_precision | train_recall | train_f1 | val_accuracy | val_precision | val_recall | val_f1 |
|-------|------------|----------------|-----------------|--------------|----------|--------------|---------------|------------|--------|
| 1     | 0.3245     | 0.8532         | 0.8612          | 0.8445       | 0.8528   | 0.8901       | 0.8956        | 0.8834     | 0.8895 |
| 2     | 0.2134     | 0.9123         | 0.9187          | 0.9056       | 0.9121   | 0.9245       | 0.9289        | 0.9198     | 0.9243 |

### Model Checkpoints

- **`best_model.pt`**: Weights from epoch with best validation accuracy
- **`checkpoint.pt`**: Latest checkpoint for resuming training

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional Backbones**: Vision Transformers (ViT), ConvNeXt, Swin Transformer
- **Ensemble Methods**: Voting ensemble combining multiple model predictions (majority vote, weighted average, stacking)
- **Data Augmentation**: Advanced augmentation strategies (RandAugment, AutoAugment)
- **Metrics**: ROC-AUC, confusion matrix visualization, per-class metrics
- **Deployment**: ONNX export, TorchScript, model quantization
- **Documentation**: Additional tutorials, use case examples

### Development Setup

```bash
git clone https://github.com/edwardyliu-uoft/distinguish.git
cd distinguish
pip install -e ".[dev]"
pytest
```

### Code Style

This project follows Python type hints and docstring conventions. Run linting:

```bash
pylint distinguish/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Related Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

---

## 💬 Support

For questions, issues, or feature requests, please [open an issue](https://github.com/edwardyliu-uoft/distinguish/issues) on GitHub.
