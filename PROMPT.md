You are an expert software engineer and ML researcher. Your task is to generate a **production-ready Python project** called `distinguish` that implements a **binary image classifier** to distinguish between **real (camera) photographs of human faces** and **AI-generated faces**.

Produce a complete, runnable codebase (file contents and directory structure) that a user can install via `pip` and run locally. Use **Python 3.12.9** and **PyTorch**. The produced project must follow best practices for packaging, documentation, modularity, and testing.

--- SPECIFICATIONS & REQUIREMENTS ---

1) Project overview and goals

- Project name / package name: `distinguish`.
- Purpose: Proof-of-concept binary classifier that labels face images as `real` or `ai_generated`.
- Design for local training and testing (limited resources); make code efficient and clear.

2) Language / platform

- Python 3.12.9.
- PyTorch (pick a stable version compatible with Python 3.12.9 — the agent may choose a suitable version).
- The project must be pip-installable (provide `pyproject.toml` or `setup.py` and other packaging metadata).

3) Core functionality (must be implemented)

- Model:
  - Build on a well-known backbone (ResNet, EfficientNet, ConvNeXt, ViT/DaViT/CoCa, etc.). Choose a backbone suitable for face images; using a pretrained backbone and fine-tuning is acceptable and encouraged.
  - Output: a single binary prediction (probability + label).
- Data handling:
  - Implement a PyTorch `Dataset` and `DataLoader` to load images from folder(s).
  - Provide clear instructions for the Kaggle dataset **“140k Real and Fake Faces”** (<https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>). Assume it is downloaded and extracted locally; show how to point CLI/config to dataset directories.
  - Include preprocessing and reasonable augmentations (resize, normalization, random flip, optional color jitter) tuned for face images.
- Training:
  - Implement training from scratch.
  - Implement fine-tuning: load weights from an external file and continue training.
  - Train and validation loops with logging of loss and accuracy. Save best model by validation metric. Allow hyperparameters via CLI/config (learning rate, epochs, batch size, optimizer choice, weight decay, etc.).
  - Support checkpoint saving and resuming.
- Model persistence:
  - Provide `save_model(path)` and `load_model(path)` utilities (use `torch.save`/`torch.load`).
- Evaluation & prediction:
  - Implement evaluation on test set with accuracy (and optionally precision/recall/F1).
  - Implement a `classify` function / CLI command that accepts one or more image file paths or a folder and outputs predicted labels and probabilities (e.g. JSON or CSV output).
- Interfaces:
  - CLI commands (via `argparse` or Click). Minimum commands:
    - `distinguish train --data <train_dir> --val <val_dir> --out <weights_path> [--epochs N] [--batch-size B] [--lr LR] [--init-weights P]`
    - `distinguish eval --data <test_dir> --weights <weights_path>`
    - `distinguish classify --weights <weights_path> <image1> <image2> ... [--out <predictions.json>]`
    - `distinguish save-model --weights <weights_path> --out <path>`
    - `distinguish load-model --weights <path>`
  - Importable API that exposes classes/functions such as:
    - `from distinguish import Trainer, Classifier, load_model, save_model, build_model, DistinguishDataset`
    - `trainer = Trainer(model, device, ...)`
    - `trainer.train(train_loader, val_loader, epochs=..., ...)`
    - `classifier = Classifier(weights_path, device)`
    - `classifier.predict(list_of_image_paths) -> list[{"path":..., "label": "real"|"ai", "score":...}]`
- Packaging:
  - Include `pyproject.toml` (PEP 621 or `setuptools`) or `setup.py`, `setup.cfg`. Include `README.md`, `LICENSE` (MIT recommended), and `CHANGELOG.md`.
  - Provide `requirements.txt` and optionally `environment.yml` (conda) listing required dependencies.
- Tests and CI:
  - Include basic unit tests for:
    - dataset loader (small synthetic sample),
    - model save/load,
    - a minimal training loop run (single batch) to ensure nothing crashes.
  - Tests may use `pytest`.
  - (Optional) provide a minimal GitHub Actions workflow for running tests.
- Logging & monitoring:
  - Use Python `logging` for messages. Save training metrics to a CSV or lightweight log (optionally support TensorBoard).
- Documentation:
  - `README.md` with:
    - short description,
    - installation instructions (`pip install .`),
    - CLI examples,
    - API examples,
    - dataset instructions (how to place Kaggle dataset folders),
    - minimal training example that runs quickly for verification (e.g., 1 epoch on a small subset).
  - Docstrings on all public classes and functions.

4) Hardware and device handling

- Detect and use GPU if available.
  - The user has an **AMD RX 9070** GPU. Attempt to use GPU if PyTorch with ROCm support is available. If ROCm is not available or not detected, gracefully fall back to CPU.
  - Also check `torch.cuda.is_available()` for CUDA-capable environments — use whichever device is supported and available.
- Provide clear logs showing which device is used.

5) Usability & robustness

- Validate inputs (existence of paths, correct types).
- Provide helpful error messages and usage examples.
- Keep code modular with sensible defaults so the project is runnable on CPU if the user lacks GPU.
- Keep memory and compute usage conservative in default configs (small batch sizes, default to fewer epochs).

6) Deliverable format (what to output)

- Return a **complete project tree** and every file content required to build and run the project. For each file include its path and full contents. Example top-level structure expectation:
  - `distinguish/` (package)
    - `__init__.py`
    - `cli.py`
    - `train.py`
    - `predict.py`
    - `model.py`
    - `data.py`
    - `utils.py`
    - `io.py` (save/load)
  - `tests/`
    - `test_data.py`
    - `test_model_io.py`
    - `test_train_smoke.py`
  - `pyproject.toml` or `setup.py` + `setup.cfg`
  - `requirements.txt`
  - `README.md`
  - `LICENSE`
  - `examples/` (small usage scripts)
- The generated code must be ready-to-run (no `TODO` placeholders).
- Provide example commands and a short example session demonstrating:
  - installing locally,
  - training for a couple of epochs on a small subset,
  - saving weights,
  - loading weights,
  - classifying a few images.

7) Dataset note

- For testing/demo, use the Kaggle dataset: `140k Real and Fake Faces` (<https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>). Explain the expected folder layout for training and validation (e.g., `train/real/`, `train/fake/`, `val/real/`, `val/fake/`) and provide code to transform the Kaggle dataset into this layout if it has a different organization.

8) Constraints & safety

- This is for a benign research demo: no actions require privileged access or external network calls. The generated project may include code that assumes the dataset is available locally; do not attempt to download the Kaggle dataset automatically without credentials.
- Do not include any copyrighted data in the response.

--- EXTRA GUIDANCE FOR THE AGENT ---

- Prefer clarity, correctness, and a working minimal default over over-optimizing for large-scale training.
- Follow Python packaging norms so `pip install .` and `python -m distinguish.cli` or an installed `distinguish` entry point works.
- Use modern PyTorch idioms: `torch.nn.Module`, `torch.utils.data.Dataset`, `DataLoader`, `optim`, mixed-precision if available (optional), checkpointing, and deterministic seeding for reproducibility.
- Provide comments and docstrings to make the code self-explanatory.

--- END ---
