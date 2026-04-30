from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from core.db import DatabaseManager
from core.metrics import compute_binary_metrics, tune_threshold
from core.model_loader import LoadedModel, NotebookAwareModelLoader
from core.reproducibility import set_global_seed


class SimpleImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform, class_to_idx: dict[str, int]) -> None:
        self.rows = rows
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        try:
            img = Image.open(row["file_path"]).convert("L")
        except Exception as exc:
            raise RuntimeError(f"Invalid image file: {row.get('file_path')}") from exc
        x = self.transform(img)
        y = self.class_to_idx[str(row["label"]).upper()]
        return x, y


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    aggregation_algorithm: str = "FedAvg"
    fedprox_mu: float = 0.0
    pretrained: bool = False
    class_weighting: bool = True
    weighted_sampler: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"
    threshold_strategy: str = "best_f1"
    min_sensitivity: float = 0.95
    seed: int = 42
    deterministic: bool = True
    img_size: int | None = None
    num_workers: int = 0
    participation_fraction: float = 1.0
    update_clip_norm: float | None = None
    dp_noise_multiplier: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class LocalTrainer:
    def __init__(self, db: DatabaseManager, loader: NotebookAwareModelLoader) -> None:
        self.db = db
        self.loader = loader
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run_async(
        self,
        loaded: LoadedModel,
        train_rows: list[dict],
        val_rows: list[dict],
        config: TrainingConfig,
        checkpoint_dir: str | Path,
        on_progress: Callable[[dict], None] | None = None,
        on_done: Callable[[dict], None] | None = None,
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self._run,
            kwargs=dict(
                loaded=loaded,
                train_rows=train_rows,
                val_rows=val_rows,
                config=config,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                on_done=on_done,
            ),
            daemon=True,
        )
        thread.start()
        return thread

    def train_sync(
        self,
        loaded: LoadedModel,
        train_rows: list[dict],
        val_rows: list[dict],
        config: TrainingConfig,
        checkpoint_dir: str | Path,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        return self._run(
            loaded=loaded,
            train_rows=train_rows,
            val_rows=val_rows,
            config=config,
            checkpoint_dir=checkpoint_dir,
            on_progress=on_progress,
            on_done=None,
        )

    def _build_transform(self, loaded: LoadedModel, train: bool = False, img_size: int | None = None):
        img_size = int(img_size or loaded.metadata.get("img_size", 320))
        common = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
        ]
        if train:
            common.extend([
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        common.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transforms.Compose(common)

    def _class_to_idx(self, rows: list[dict]) -> dict[str, int]:
        labels = {str(r["label"]).upper() for r in rows}
        if labels and labels.issubset({"NORMAL", "PNEUMONIA"}):
            return {"NORMAL": 0, "PNEUMONIA": 1}
        ordered = sorted(labels)
        if len(ordered) != 2:
            raise ValueError(
                "Binary training expects exactly two classes. Expected NORMAL and PNEUMONIA folders/labels."
            )
        return {ordered[0]: 0, ordered[1]: 1}

    def _build_sampler(self, rows: list[dict], class_to_idx: dict[str, int]):
        counts: dict[int, int] = {}
        targets = []
        for row in rows:
            target = class_to_idx[str(row["label"]).upper()]
            targets.append(target)
            counts[target] = counts.get(target, 0) + 1
        sample_weights = [1.0 / max(counts[target], 1) for target in targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def _pos_weight(self, rows: list[dict], class_to_idx: dict[str, int], device) -> torch.Tensor | None:
        counts = {0: 0, 1: 0}
        for row in rows:
            counts[class_to_idx[str(row["label"]).upper()]] += 1
        if counts[1] == 0:
            return None
        return torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device)

    def _run(
        self,
        loaded: LoadedModel,
        train_rows: list[dict],
        val_rows: list[dict],
        config: TrainingConfig,
        checkpoint_dir: str | Path,
        on_progress=None,
        on_done=None,
    ) -> dict:
        self._stop_requested = False
        set_global_seed(config.seed, config.deterministic)

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not train_rows:
            raise ValueError("Training cannot start with an empty training split.")

        class_to_idx = self._class_to_idx(train_rows + val_rows)
        train_transform = self._build_transform(loaded, train=True, img_size=config.img_size)
        eval_transform = self._build_transform(loaded, train=False, img_size=config.img_size)
        train_ds = SimpleImageDataset(train_rows, train_transform, class_to_idx)
        val_ds = SimpleImageDataset(val_rows, eval_transform, class_to_idx)

        sampler = self._build_sampler(train_rows, class_to_idx) if config.weighted_sampler else None
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = loaded.model.to(self.loader.device)
        device = self.loader.device

        pos_weight = self._pos_weight(train_rows, class_to_idx, device) if config.class_weighting else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        global_state = {name: param.detach().clone().to(device) for name, param in model.named_parameters()}
        best_value = None
        best_path = checkpoint_dir / "local_best.pt"
        history = []
        no_improve_epochs = 0
        threshold = float(getattr(loaded, "threshold", loaded.metadata.get("threshold", 0.5)))

        for epoch in range(1, config.epochs + 1):
            if self._stop_requested:
                break

            model.train()
            total_loss = 0.0
            y_true: list[int] = []
            y_prob: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).float().view(-1)

                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)

                if config.aggregation_algorithm.lower() == "fedprox" and config.fedprox_mu > 0:
                    prox = torch.zeros((), device=device)
                    for name, param in model.named_parameters():
                        prox = prox + torch.sum((param - global_state[name]) ** 2)
                    loss = loss + (config.fedprox_mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(logits.detach()).view(-1)
                bs = xb.size(0)
                total_loss += loss.item() * bs
                y_true.extend(yb.long().detach().cpu().tolist())
                y_prob.extend(probs.detach().cpu().tolist())

            train_loss = total_loss / max(len(y_true), 1)
            train_metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
            train_metrics["loss"] = train_loss

            val_metrics = self._evaluate(
                model,
                val_loader,
                criterion,
                device,
                fallback_threshold=threshold,
                threshold_strategy=config.threshold_strategy,
                min_sensitivity=config.min_sensitivity,
            )
            threshold = float(val_metrics.get("threshold", threshold))

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1_score": val_metrics["f1_score"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "val_false_negatives": val_metrics["false_negatives"],
                "val_false_positives": val_metrics["false_positives"],
                "threshold": threshold,
            }
            history.append(row)

            improved, metric_value = self._is_improved(config.early_stopping_metric, val_metrics, best_value)
            if improved:
                best_value = metric_value
                no_improve_epochs = 0
                torch.save(
                    self._checkpoint_payload(loaded, model, threshold, val_metrics, config),
                    best_path,
                )
            else:
                no_improve_epochs += 1

            if on_progress:
                on_progress(row)

            if config.early_stopping_patience > 0 and no_improve_epochs >= config.early_stopping_patience:
                break

            time.sleep(0.05)

        best_metrics = history[-1] if history else {}
        if best_path.exists():
            try:
                checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                best_metrics = checkpoint.get("metrics", best_metrics)
            except Exception:
                self.db.log("training", f"Best checkpoint exists but could not be read: {best_path}", "warning")

        result = {
            "best_accuracy": float(best_metrics.get("accuracy", best_metrics.get("val_accuracy", 0.0))),
            "best_loss": float(best_metrics.get("loss", best_metrics.get("val_loss", 0.0))),
            "best_metrics": best_metrics,
            "best_path": str(best_path),
            "history": history,
            "num_samples": len(train_rows),
            "threshold": float(best_metrics.get("threshold", threshold)),
            "training_config": config.to_dict(),
        }

        self.db.save_model_version(
            model_name=loaded.metadata.get("model_name", "torchvision.densenet121"),
            architecture=loaded.metadata.get("architecture", "torchvision.densenet121_binary"),
            version=best_path.stem,
            file_path=str(best_path),
            source="local_training",
            aggregation_algorithm=config.aggregation_algorithm,
            threshold=result["threshold"],
            metrics=best_metrics,
            training_config=config.to_dict(),
            metadata={"class_names": loaded.class_names, "checkpoint_path": loaded.checkpoint_path},
        )
        self.db.log(
            "training",
            f"Completed local training with best_acc={result['best_accuracy']:.4f}, threshold={result['threshold']:.3f}",
            "success",
        )

        if on_done:
            on_done(result)
        return result

    def _is_improved(self, metric_name: str, val_metrics: dict, best_value):
        metric_name = (metric_name or "val_loss").lower()
        if metric_name in {"val_loss", "loss"}:
            value = float(val_metrics.get("loss", float("inf")))
            return best_value is None or value < best_value, value
        key = {
            "val_accuracy": "accuracy",
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "val_roc_auc": "roc_auc",
            "f1": "f1_score",
            "f1_score": "f1_score",
            "sensitivity": "sensitivity",
        }.get(metric_name, "accuracy")
        value = val_metrics.get(key)
        value = -1.0 if value is None else float(value)
        return best_value is None or value > best_value, value

    def _checkpoint_payload(
        self,
        loaded: LoadedModel,
        model,
        threshold: float,
        metrics: dict,
        config: TrainingConfig,
    ) -> dict:
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return {
            "state_dict": state_dict,
            "target_label": loaded.metadata.get("target_label", "PNEUMONIA"),
            "threshold": float(threshold),
            "img_size": int(config.img_size or loaded.metadata.get("img_size", 320)),
            "model_name": "torchvision.densenet121",
            "architecture": "torchvision.densenet121_binary",
            "class_names": loaded.class_names or ["NORMAL", "PNEUMONIA"],
            "classes": loaded.class_names or ["NORMAL", "PNEUMONIA"],
            "mask_mode": loaded.metadata.get("mask_mode"),
            "metrics": metrics,
            "training_config": config.to_dict(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _evaluate(
        self,
        model,
        loader,
        criterion,
        device,
        fallback_threshold: float,
        threshold_strategy: str = "best_f1",
        min_sensitivity: float = 0.95,
    ) -> dict:
        model.eval()
        total_loss = 0.0
        y_true: list[int] = []
        y_prob: list[float] = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device).float().view(-1)

                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits).view(-1)

                bs = xb.size(0)
                total_loss += loss.item() * bs
                y_true.extend(yb.long().cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        if threshold_strategy in {"fixed", "fixed_0_5", "0.5"}:
            metrics = compute_binary_metrics(y_true, y_prob, threshold=fallback_threshold)
        else:
            selection = tune_threshold(
                y_true,
                y_prob,
                strategy=threshold_strategy,
                min_sensitivity=min_sensitivity,
            )
            metrics = selection.metrics
            metrics["threshold_strategy"] = selection.strategy
        metrics["loss"] = total_loss / max(len(y_true), 1)
        return metrics
