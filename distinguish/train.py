"""Training utilities and Trainer class."""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from .utils import (
    get_device,
    seed_everything,
    compute_binary_metrics,
    write_metrics_csv,
    ensure_dir,
)

logger = logging.getLogger("distinguish")


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""

    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 5
    weight_decay: float = 0.0
    optimizer: str = "adam"
    mixed_precision: bool = False
    out_dir: str = "runs"
    checkpoint_path: Optional[str] = None
    resume: bool = False
    grad_clip: Optional[float] = None
    seed: int = 42


class Trainer:
    """Encapsulates training and evaluation loops.

    Parameters
    ----------
    model: torch.nn.Module
        The model to train.
    device: torch.device
        Device on which to run training.
    config: TrainerConfig
        Training hyperparameters and options.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.device = device or get_device()
        self.config = config or TrainerConfig()
        seed_everything(self.config.seed)
        self.model.to(self.device)
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        self.criterion = nn.BCEWithLogitsLoss()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            device_type,
            enabled=self.config.mixed_precision and self.device.type == "cuda",
        )
        ensure_dir(self.config.out_dir)
        self.best_metric = -1.0
        self.start_epoch = 0
        checkpoint_exists = (
            self.config.resume
            and self.config.checkpoint_path
            and os.path.isfile(self.config.checkpoint_path)
        )
        if checkpoint_exists:
            self._load_checkpoint(self.config.checkpoint_path)

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        path = os.path.join(self.config.out_dir, "checkpoint.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
            },
            path,
        )
        logger.info("Saved checkpoint: %s (metric=%.4f)", path, metric)

    def _load_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model_state"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(data["optimizer_state"])  # type: ignore[arg-type]
        self.best_metric = data.get("best_metric", -1.0)
        self.start_epoch = data.get("epoch", 0) + 1
        logger.info("Resumed from checkpoint %s at epoch %s", path, self.start_epoch)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : Optional[DataLoader]
            Validation data loader.

        Returns
        -------
        Dict[str, Any]
            Training history and best metric.
        """
        history = []
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            preds_all = []
            targets_all = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch in pbar:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type, enabled=self.scaler.is_enabled()):
                    logits = self.model(images).view(-1)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item() * images.size(0)
                preds_all.append(logits.detach().cpu())
                targets_all.append(labels.detach().cpu())
            train_logits = torch.cat(preds_all)
            train_targets = torch.cat(targets_all)
            train_metrics = compute_binary_metrics(train_logits, train_targets)
            train_loss = epoch_loss / len(train_loader.dataset)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metric_to_compare = val_metrics.get("accuracy", 0.0)
                if metric_to_compare > self.best_metric:
                    self.best_metric = metric_to_compare
                    best_path = os.path.join(self.config.out_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)
                    logger.info(
                        "New best model saved (accuracy=%.4f)",
                        metric_to_compare,
                    )
                    self._save_checkpoint(epoch, metric_to_compare)

            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(record)
            logger.info(
                "Epoch %d: train_loss=%.4f train_acc=%.4f val_acc=%.4f",
                epoch + 1,
                train_loss,
                train_metrics["accuracy"],
                val_metrics.get("accuracy", 0.0),
            )
        metrics_path = os.path.join(self.config.out_dir, "metrics.csv")
        fieldnames = list(history[0].keys()) if history else []
        if history:
            write_metrics_csv(metrics_path, history, fieldnames)
            logger.info("Wrote metrics CSV: %s", metrics_path)
        return {"history": history, "best_metric": self.best_metric}

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader for evaluation.

        Returns
        -------
        Dict[str, float]
            Computed metrics.
        """
        self.model.eval()
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images).view(-1)
                preds_all.append(logits.cpu())
                targets_all.append(labels.cpu())
        logits = torch.cat(preds_all)
        targets = torch.cat(targets_all)
        return compute_binary_metrics(logits, targets)
