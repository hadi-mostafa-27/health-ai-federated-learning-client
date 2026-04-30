from __future__ import annotations

import copy
import random
import threading
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.metrics import compute_binary_metrics, tune_threshold
from core.model_loader import LoadedModel
from core.trainer import LocalTrainer, SimpleImageDataset, TrainingConfig


class FederatedEngine:
    def __init__(self, global_loaded: LoadedModel, config: TrainingConfig, device: str):
        self.global_loaded = global_loaded
        self.config = config
        self.device = device

        self.global_model = global_loaded.model.to(device)
        self.client_models: Dict[str, nn.Module] = {}
        self.client_loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}
        self.client_sample_counts: Dict[str, int] = {}
        self.client_updates: Dict[str, dict] = {}
        self.round_history: list[dict] = []
        self.global_metrics: list[dict] = []
        self.current_round = 0

    @property
    def algorithm(self) -> str:
        configured = getattr(self.config, "aggregation_algorithm", None) or "FedAvg"
        return "FedProx" if configured.lower() == "fedprox" else "FedAvg"

    def distribute_initial_model(self, hospitals: List[str]):
        """Server broadcasts the same starting global model to selected hospitals."""
        self.client_models = {}
        for hospital_id in hospitals:
            self.client_models[hospital_id] = copy.deepcopy(self.global_model).to(self.device)

    def prepare_datasets(self, datasets: Dict[str, List[dict]], trainer: LocalTrainer):
        """Prepare deterministic local train/validation loaders for each hospital."""
        self.client_loaders = {}
        self.client_sample_counts = {}
        train_transform = trainer._build_transform(self.global_loaded, train=True, img_size=self.config.img_size)
        eval_transform = trainer._build_transform(self.global_loaded, train=False, img_size=self.config.img_size)

        for hospital_id, rows in datasets.items():
            if not rows:
                continue
            class_to_idx = trainer._class_to_idx(rows)
            train_rows, val_rows = self._split_rows(rows, seed=self.config.seed)
            train_ds = SimpleImageDataset(train_rows, train_transform, class_to_idx)
            val_ds = SimpleImageDataset(val_rows, eval_transform, class_to_idx)

            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

            self.client_loaders[hospital_id] = (train_loader, val_loader)
            self.client_sample_counts[hospital_id] = len(train_rows)

    def _split_rows(self, rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
        if len(rows) < 2:
            return list(rows), []
        labels = [str(row["label"]).upper() for row in rows]
        stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) >= 2 else None
        try:
            train_rows, val_rows = train_test_split(
                rows,
                test_size=0.2,
                random_state=seed,
                stratify=stratify,
            )
            return list(train_rows), list(val_rows)
        except ValueError:
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            cut = max(1, int(0.8 * len(shuffled)))
            return shuffled[:cut], shuffled[cut:]

    def select_participants(
        self,
        hospitals: list[str] | None = None,
        participation_fraction: float | None = None,
        seed: int | None = None,
        dropped_clients: set[str] | None = None,
    ) -> list[str]:
        candidates = hospitals or list(self.client_loaders.keys())
        candidates = [h for h in candidates if h in self.client_loaders and h in self.client_models]
        dropped_clients = dropped_clients or set()
        candidates = [h for h in candidates if h not in dropped_clients]
        if not candidates:
            return []

        fraction = participation_fraction if participation_fraction is not None else self.config.participation_fraction
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0:
            return []
        target = max(1, int(round(len(candidates) * fraction)))
        rng = random.Random(self.config.seed if seed is None else seed)
        return sorted(rng.sample(candidates, min(target, len(candidates))))

    def run_local_training_async(self, hospital_id: str, on_progress: Callable, on_done: Callable) -> threading.Thread:
        """Train one hospital's model locally and record a complete client update."""
        global_reference = {
            name: param.detach().clone().to(self.device)
            for name, param in self.global_model.named_parameters()
        }

        def _train_worker():
            try:
                result = self._train_client(hospital_id, global_reference, on_progress)
                self.client_updates[hospital_id] = result
                if on_done:
                    on_done(result)
            except Exception as exc:
                result = {
                    "hospital_id": hospital_id,
                    "status": "failed",
                    "error": str(exc),
                    "num_samples": self.client_sample_counts.get(hospital_id, 0),
                }
                self.client_updates[hospital_id] = result
                if on_done:
                    on_done(result)

        thread = threading.Thread(target=_train_worker, daemon=True)
        thread.start()
        return thread

    def _train_client(self, hospital_id: str, global_reference: dict[str, torch.Tensor], on_progress=None) -> dict:
        if hospital_id not in self.client_models:
            raise KeyError(f"No local model was distributed to {hospital_id}.")
        if hospital_id not in self.client_loaders:
            raise KeyError(f"No local dataset loader is available for {hospital_id}.")

        model = self.client_models[hospital_id]
        train_loader, val_loader = self.client_loaders[hospital_id]
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        threshold = float(getattr(self.global_loaded, "threshold", self.global_loaded.metadata.get("threshold", 0.5)))
        latest_train_metrics: dict = {}
        latest_val_metrics: dict = {}

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_loss = 0.0
            y_true: list[int] = []
            y_prob: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).float().view(-1)

                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)

                if self.algorithm == "FedProx" and self.config.fedprox_mu > 0:
                    prox = torch.zeros((), device=self.device)
                    for name, param in model.named_parameters():
                        prox = prox + torch.sum((param - global_reference[name]) ** 2)
                    loss = loss + (self.config.fedprox_mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(logits.detach()).view(-1)
                total_loss += loss.item() * xb.size(0)
                y_true.extend(yb.long().detach().cpu().tolist())
                y_prob.extend(probs.detach().cpu().tolist())

            latest_train_metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
            latest_train_metrics["loss"] = total_loss / max(len(y_true), 1)
            latest_val_metrics = self._evaluate_loader(model, val_loader, criterion, threshold)
            threshold = float(latest_val_metrics.get("threshold", threshold))

            if on_progress:
                on_progress({
                    "hospital_id": hospital_id,
                    "epoch": epoch,
                    "loss": latest_train_metrics["loss"],
                    "acc": latest_train_metrics["accuracy"],
                    "val_loss": latest_val_metrics["loss"],
                    "val_acc": latest_val_metrics["accuracy"],
                    "threshold": threshold,
                    "false_negatives": latest_val_metrics["false_negatives"],
                })

        return {
            "hospital_id": hospital_id,
            "status": "completed",
            "num_samples": self.client_sample_counts.get(hospital_id, 0),
            "local_loss": latest_train_metrics.get("loss"),
            "local_accuracy": latest_train_metrics.get("accuracy"),
            "local_metrics": latest_train_metrics,
            "validation_metrics": latest_val_metrics,
            "threshold": threshold,
        }

    def _evaluate_loader(self, model, loader, criterion, fallback_threshold: float) -> dict:
        model.eval()
        total_loss = 0.0
        y_true: list[int] = []
        y_prob: list[float] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).float().view(-1)
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits).view(-1)
                total_loss += loss.item() * xb.size(0)
                y_true.extend(yb.long().cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        if getattr(self.config, "threshold_strategy", "best_f1") in {"fixed", "fixed_0_5", "0.5"}:
            metrics = compute_binary_metrics(y_true, y_prob, threshold=fallback_threshold)
        else:
            selection = tune_threshold(
                y_true,
                y_prob,
                strategy=getattr(self.config, "threshold_strategy", "best_f1"),
                min_sensitivity=getattr(self.config, "min_sensitivity", 0.95),
            )
            metrics = selection.metrics
            metrics["threshold_strategy"] = selection.strategy
        metrics["loss"] = total_loss / max(len(y_true), 1)
        return metrics

    def aggregate_models(
        self,
        participating_clients: list[str] | None = None,
        algorithm: str | None = None,
        round_number: int | None = None,
    ) -> dict:
        """Aggregate local models with sample-weighted FedAvg/FedProx updates."""
        algorithm = algorithm or self.algorithm
        participating_clients = participating_clients or list(self.client_updates.keys())
        completed = [
            client_id
            for client_id in participating_clients
            if self.client_updates.get(client_id, {}).get("status") == "completed"
            and self.client_sample_counts.get(client_id, 0) > 0
        ]
        if not completed:
            raise RuntimeError("Cannot aggregate: zero completed client updates.")

        total_samples = sum(self.client_sample_counts[client_id] for client_id in completed)
        if total_samples <= 0:
            raise RuntimeError("Cannot aggregate: participating clients report zero samples.")

        base_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        client_states = {
            client_id: self._privacy_adjusted_state(self.client_models[client_id].state_dict(), base_state)
            for client_id in completed
        }
        new_state = copy.deepcopy(base_state)

        for key, global_tensor in base_state.items():
            if not torch.is_floating_point(global_tensor):
                new_state[key] = client_states[completed[0]][key].to(global_tensor.dtype)
                continue
            aggregated = torch.zeros_like(global_tensor, dtype=torch.float32)
            for client_id in completed:
                weight = self.client_sample_counts[client_id] / total_samples
                aggregated = aggregated + client_states[client_id][key].float().to(aggregated.device) * weight
            new_state[key] = aggregated.to(global_tensor.dtype)

        self.global_model.load_state_dict(new_state)
        self.global_loaded.model = self.global_model

        record = {
            "round_number": round_number if round_number is not None else self.current_round,
            "aggregation_algorithm": algorithm,
            "participating_clients": completed,
            "client_sample_counts": {client_id: self.client_sample_counts[client_id] for client_id in completed},
            "total_samples": total_samples,
            "client_updates": {client_id: self.client_updates[client_id] for client_id in completed},
            "weighted_fedavg_formula": "sum((client_samples / total_samples) * client_weights)",
            "dp_noise_multiplier": getattr(self.config, "dp_noise_multiplier", 0.0),
            "update_clip_norm": getattr(self.config, "update_clip_norm", None),
        }
        self.round_history.append(record)
        return record

    def _privacy_adjusted_state(self, client_state: dict, base_state: dict) -> dict:
        clip_norm = getattr(self.config, "update_clip_norm", None)
        noise_multiplier = float(getattr(self.config, "dp_noise_multiplier", 0.0) or 0.0)
        if not clip_norm and noise_multiplier <= 0:
            return client_state

        squared_norm = torch.zeros(())
        for key, base_tensor in base_state.items():
            if torch.is_floating_point(base_tensor):
                delta = client_state[key].detach().cpu().float() - base_tensor.detach().cpu().float()
                squared_norm = squared_norm + torch.sum(delta ** 2)
        update_norm = torch.sqrt(squared_norm).item()
        scale = 1.0
        if clip_norm and update_norm > 0:
            scale = min(1.0, float(clip_norm) / update_norm)

        adjusted = {}
        for key, base_tensor in base_state.items():
            value = client_state[key]
            if torch.is_floating_point(base_tensor):
                delta = (value.detach().cpu().float() - base_tensor.detach().cpu().float()) * scale
                if noise_multiplier > 0:
                    noise_std = float(clip_norm or 1.0) * noise_multiplier
                    delta = delta + torch.normal(mean=0.0, std=noise_std, size=delta.shape)
                adjusted[key] = (base_tensor.detach().cpu().float() + delta).to(value.device).to(value.dtype)
            else:
                adjusted[key] = value
        return adjusted

    def evaluate_global_model(self, trainer: LocalTrainer | None = None, round_number: int | None = None) -> dict:
        """Evaluate the current global model on all available validation loaders."""
        self.global_model.eval()
        criterion = nn.BCEWithLogitsLoss()
        threshold = float(getattr(self.global_loaded, "threshold", self.global_loaded.metadata.get("threshold", 0.5)))

        total_loss = 0.0
        y_true: list[int] = []
        y_prob: list[float] = []

        with torch.no_grad():
            for _, val_loader in self.client_loaders.values():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).float().view(-1)
                    logits = self.global_model(xb).view(-1)
                    loss = criterion(logits, yb)
                    probs = torch.sigmoid(logits).view(-1)
                    total_loss += loss.item() * xb.size(0)
                    y_true.extend(yb.long().cpu().tolist())
                    y_prob.extend(probs.cpu().tolist())

        if getattr(self.config, "threshold_strategy", "best_f1") in {"fixed", "fixed_0_5", "0.5"}:
            metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        else:
            selection = tune_threshold(
                y_true,
                y_prob,
                strategy=getattr(self.config, "threshold_strategy", "best_f1"),
                min_sensitivity=getattr(self.config, "min_sensitivity", 0.95),
            )
            metrics = selection.metrics
            metrics["threshold_strategy"] = selection.strategy

        metrics["loss"] = total_loss / max(len(y_true), 1)
        metrics["round_number"] = round_number if round_number is not None else self.current_round
        self.global_loaded.threshold = float(metrics["threshold"])
        self.global_loaded.metadata["threshold"] = float(metrics["threshold"])
        self.global_metrics.append(metrics)
        return metrics

    def save_global_checkpoint(self, output_path: str | Path, metrics: dict | None = None) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()},
                "threshold": float(self.global_loaded.threshold),
                "img_size": self.global_loaded.metadata.get("img_size", self.config.img_size or 320),
                "model_name": "torchvision.densenet121",
                "architecture": "torchvision.densenet121_binary",
                "class_names": self.global_loaded.class_names or ["NORMAL", "PNEUMONIA"],
                "classes": self.global_loaded.class_names or ["NORMAL", "PNEUMONIA"],
                "aggregation_algorithm": self.algorithm,
                "metrics": metrics or (self.global_metrics[-1] if self.global_metrics else {}),
                "training_config": self.config.to_dict(),
                "round_history": self.round_history,
            },
            output_path,
        )
        return str(output_path)
