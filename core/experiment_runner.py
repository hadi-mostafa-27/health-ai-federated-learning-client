from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.dataset_manager import DatasetManager
from core.db import DatabaseManager
from core.fl_engine import FederatedEngine
from core.metrics import compute_binary_metrics
from core.model_loader import LoadedModel, NotebookAwareModelLoader
from core.non_iid import FederatedSplitConfig, split_federated_rows, summarize_federated_split
from core.report_generator import ReportGenerator
from core.reproducibility import collect_environment_info, create_run_id, export_experiment_config, set_global_seed
from core.secure_aggregation import communication_size_bytes
from core.trainer import LocalTrainer, SimpleImageDataset, TrainingConfig


@dataclass
class ExperimentConfig:
    run_name: str = "pneumonia_fl_comparison"
    methods: list[str] = field(default_factory=lambda: ["local", "centralized", "fedavg", "fedprox"])
    dataset_id: int | None = None
    seed: int = 42
    num_hospitals: int = 3
    non_iid_strategy: str = "balanced_iid"
    imbalance_severity: float = 0.5
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    participation_fraction: float = 1.0
    fedprox_mu: float = 0.01
    threshold_strategy: str = "best_f1"
    min_sensitivity: float = 0.95
    pretrained: bool = False
    output_dir: str = "reports/experiments"
    security_mode: str = "none"


class ExperimentRunner:
    def __init__(
        self,
        db: DatabaseManager,
        loader: NotebookAwareModelLoader,
        dataset_manager: DatasetManager,
        reporter: ReportGenerator | None = None,
    ) -> None:
        self.db = db
        self.loader = loader
        self.dataset_manager = dataset_manager
        self.reporter = reporter or ReportGenerator(db)
        self.trainer = LocalTrainer(db, loader)

    def run(self, config: ExperimentConfig) -> dict[str, Any]:
        set_global_seed(config.seed, deterministic=True)
        run_id = create_run_id(config.run_name)
        cfg_dict = asdict(config)
        environment = collect_environment_info()
        output_dir = Path(config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        export_experiment_config(cfg_dict, output_dir, run_id)

        dataset_id = config.dataset_id
        if dataset_id is None:
            latest = self.dataset_manager.latest_dataset()
            if not latest:
                raise RuntimeError("Register a dataset before running experiments.")
            dataset_id = int(latest["id"])

        rows = {
            "train": self.dataset_manager.images_for_split(dataset_id, "train"),
            "val": self.dataset_manager.images_for_split(dataset_id, "val"),
            "test": self.dataset_manager.images_for_split(dataset_id, "test"),
        }
        if not rows["train"] or not rows["test"]:
            raise RuntimeError("Experiment requires non-empty train and test splits.")

        self.db.create_experiment_run(
            run_id=run_id,
            run_name=config.run_name,
            experiment_type="comparison",
            aggregation_algorithm="multiple",
            dataset_id=dataset_id,
            seed=config.seed,
            config=cfg_dict,
            environment=environment,
        )

        summary_rows: list[dict[str, Any]] = []
        per_round_rows: list[dict[str, Any]] = []
        client_rows: list[dict[str, Any]] = []
        artifacts: dict[str, Any] = {}

        try:
            methods = [method.lower() for method in config.methods]
            if "local" in methods or "local-only" in methods:
                local_result = self._run_local_only(run_id, dataset_id, rows, config, output_dir)
                summary_rows.extend(local_result["summary_rows"])
                client_rows.extend(local_result["client_level"])
                artifacts["local"] = local_result["artifacts"]

            if "centralized" in methods:
                central_result = self._run_centralized(run_id, dataset_id, rows, config, output_dir)
                summary_rows.append(central_result["summary"])
                artifacts["centralized"] = central_result["artifacts"]

            if "fedavg" in methods:
                fedavg_result = self._run_federated(run_id, dataset_id, rows, config, output_dir, "FedAvg")
                summary_rows.append(fedavg_result["summary"])
                per_round_rows.extend(fedavg_result["per_round"])
                client_rows.extend(fedavg_result["client_level"])
                artifacts["fedavg"] = fedavg_result["artifacts"]

            if "fedprox" in methods:
                fedprox_result = self._run_federated(run_id, dataset_id, rows, config, output_dir, "FedProx")
                summary_rows.append(fedprox_result["summary"])
                per_round_rows.extend(fedprox_result["per_round"])
                client_rows.extend(fedprox_result["client_level"])
                artifacts["fedprox"] = fedprox_result["artifacts"]

            report_payload = {
                "run_id": run_id,
                "config": cfg_dict,
                "environment": environment,
                "summary_rows": summary_rows,
                "per_round": per_round_rows,
                "client_level": client_rows,
                "artifacts": artifacts,
            }
            for row in summary_rows + per_round_rows + client_rows:
                row.setdefault("security_mode", config.security_mode)
                row.setdefault("number_of_clients", config.num_hospitals)
            report_paths = self.reporter.save_experiment_report(run_id, report_payload)
            report_payload["report_paths"] = report_paths
            self.db.finish_experiment_run(run_id, "completed", report_payload)
            return report_payload
        except Exception as exc:
            self.db.finish_experiment_run(run_id, "failed", {"error": str(exc)})
            raise

    def _training_config(self, config: ExperimentConfig, algorithm: str) -> TrainingConfig:
        return TrainingConfig(
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            aggregation_algorithm=algorithm,
            fedprox_mu=config.fedprox_mu if algorithm.lower() == "fedprox" else 0.0,
            pretrained=config.pretrained,
            seed=config.seed,
            participation_fraction=config.participation_fraction,
            threshold_strategy=config.threshold_strategy,
            min_sensitivity=config.min_sensitivity,
        )

    def _fresh_loaded(self, pretrained: bool = False) -> LoadedModel:
        model = self.loader.build_model(pretrained=pretrained)
        return LoadedModel(
            model=model,
            checkpoint_path="initialized",
            num_classes=1,
            class_names=["NORMAL", "PNEUMONIA"],
            notebook_profile="initialized",
            metadata={
                "model_name": "torchvision.densenet121",
                "architecture": "torchvision.densenet121_binary",
                "img_size": 224,
                "threshold": 0.5,
                "target_label": "PNEUMONIA",
            },
            threshold=0.5,
            checkpoint=None,
        )

    def _run_local_only(
        self,
        run_id: str,
        dataset_id: int,
        rows: dict[str, list[dict]],
        config: ExperimentConfig,
        output_dir: Path,
    ) -> dict:
        split = split_federated_rows(
            rows["train"],
            FederatedSplitConfig(
                strategy=config.non_iid_strategy,
                num_hospitals=config.num_hospitals,
                seed=config.seed,
                imbalance_severity=config.imbalance_severity,
            ),
        )
        self._persist_split_distribution(dataset_id, run_id, config.non_iid_strategy, split, "train")

        summary_rows = []
        client_level = []
        artifacts = {}
        for hospital_id, train_rows in split.items():
            if not train_rows:
                continue
            loaded = self._fresh_loaded(config.pretrained)
            result = self.trainer.train_sync(
                loaded,
                train_rows,
                rows["val"],
                self._training_config(config, "LocalOnly"),
                output_dir / "local_only" / hospital_id,
            )
            loaded.threshold = float(result["threshold"])
            loaded.metadata["threshold"] = loaded.threshold
            metrics = self._evaluate_rows(loaded, rows["test"], loaded.threshold)
            row = self._summary_row("local_only", metrics, hospital_id=hospital_id)
            summary_rows.append(row)
            client_level.append(row)
            artifacts[hospital_id] = result["best_path"]
            self._persist_metrics(run_id, "local_only", metrics, hospital_id=hospital_id, split="test")
        return {"summary_rows": summary_rows, "client_level": client_level, "artifacts": artifacts}

    def _run_centralized(
        self,
        run_id: str,
        dataset_id: int,
        rows: dict[str, list[dict]],
        config: ExperimentConfig,
        output_dir: Path,
    ) -> dict:
        loaded = self._fresh_loaded(config.pretrained)
        result = self.trainer.train_sync(
            loaded,
            rows["train"],
            rows["val"],
            self._training_config(config, "Centralized"),
            output_dir / "centralized",
        )
        loaded.threshold = float(result["threshold"])
        loaded.metadata["threshold"] = loaded.threshold
        metrics = self._evaluate_rows(loaded, rows["test"], loaded.threshold)
        summary = self._summary_row("centralized", metrics)
        self._persist_metrics(run_id, "centralized", metrics, split="test")
        return {"summary": summary, "artifacts": {"checkpoint": result["best_path"]}}

    def _run_federated(
        self,
        run_id: str,
        dataset_id: int,
        rows: dict[str, list[dict]],
        config: ExperimentConfig,
        output_dir: Path,
        algorithm: str,
    ) -> dict:
        split = split_federated_rows(
            rows["train"],
            FederatedSplitConfig(
                strategy=config.non_iid_strategy,
                num_hospitals=config.num_hospitals,
                seed=config.seed,
                imbalance_severity=config.imbalance_severity,
            ),
        )
        self._persist_split_distribution(dataset_id, run_id, config.non_iid_strategy, split, "train")

        loaded = self._fresh_loaded(config.pretrained)
        train_cfg = self._training_config(config, algorithm)
        engine = FederatedEngine(loaded, train_cfg, self.loader.device)
        hospitals = [hid for hid, hospital_rows in split.items() if hospital_rows]
        engine.distribute_initial_model(hospitals)
        engine.prepare_datasets(split, self.trainer)

        per_round = []
        client_level = []
        best_metrics = None
        best_checkpoint = output_dir / algorithm.lower() / f"{algorithm.lower()}_best.pt"

        for round_number in range(1, config.rounds + 1):
            engine.current_round = round_number
            engine.client_updates = {}
            participants = engine.select_participants(
                hospitals,
                participation_fraction=config.participation_fraction,
                seed=config.seed + round_number,
            )
            if not participants:
                raise RuntimeError(f"Round {round_number} has zero participating clients.")

            threads = [
                engine.run_local_training_async(hospital_id, on_progress=None, on_done=None)
                for hospital_id in participants
            ]
            for thread in threads:
                thread.join()

            for hospital_id in participants:
                update = engine.client_updates.get(hospital_id, {})
                self.db.save_client_update(
                    experiment_run_id=run_id,
                    round_number=round_number,
                    hospital_id=hospital_id,
                    num_samples=int(update.get("num_samples", 0)),
                    local_loss=update.get("local_loss"),
                    local_accuracy=update.get("local_accuracy"),
                    local_metrics=update.get("validation_metrics") or update.get("local_metrics"),
                    status=update.get("status", "missing"),
                )
                if update.get("status") == "completed":
                    metrics = update.get("validation_metrics") or {}
                    client_level.append(self._summary_row(algorithm.lower(), metrics, round_number, hospital_id))

            communication_size = sum(
                communication_size_bytes(engine.client_models[hospital_id].state_dict())
                for hospital_id in participants
                if hospital_id in engine.client_models
            )
            aggregation_start = perf_counter()
            aggregation_record = engine.aggregate_models(participants, algorithm=algorithm, round_number=round_number)
            aggregation_time = perf_counter() - aggregation_start
            global_metrics = engine.evaluate_global_model(round_number=round_number)
            global_metrics.update({
                "security_mode": config.security_mode,
                "number_of_clients": len(hospitals),
                "number_of_completed_clients": len(aggregation_record["participating_clients"]),
                "dropped_clients": sorted(set(hospitals) - set(aggregation_record["participating_clients"])),
                "aggregation_time_seconds": aggregation_time,
                "communication_size_bytes": communication_size,
            })
            per_round.append(self._summary_row(algorithm.lower(), global_metrics, round_number))
            self._persist_metrics(run_id, algorithm.lower(), global_metrics, round_number=round_number, split="validation")
            self.db.save_federated_round(
                experiment_run_id=run_id,
                round_number=round_number,
                aggregation_algorithm=algorithm,
                participation_fraction=config.participation_fraction,
                participating_clients=aggregation_record["participating_clients"],
                client_sample_counts=aggregation_record["client_sample_counts"],
                global_metrics=global_metrics,
                status="completed",
            )
            engine.distribute_initial_model(hospitals)

            if best_metrics is None or (global_metrics.get("f1_score", 0) > best_metrics.get("f1_score", 0)):
                best_metrics = global_metrics
                engine.save_global_checkpoint(best_checkpoint, metrics=global_metrics)

        test_metrics = self._evaluate_rows(loaded, rows["test"], float(loaded.threshold))
        summary = self._summary_row(algorithm.lower(), test_metrics)
        self._persist_metrics(run_id, algorithm.lower(), test_metrics, split="test")
        self.db.save_model_version(
            model_name="torchvision.densenet121",
            architecture="torchvision.densenet121_binary",
            version=best_checkpoint.stem,
            file_path=str(best_checkpoint),
            source="federated_experiment",
            aggregation_algorithm=algorithm,
            threshold=float(loaded.threshold),
            metrics=test_metrics,
            training_config=train_cfg.to_dict(),
            metadata={"run_id": run_id, "rounds": config.rounds},
        )
        return {
            "summary": summary,
            "per_round": per_round,
            "client_level": client_level,
            "artifacts": {
                "checkpoint": str(best_checkpoint),
                "split_summary": summarize_federated_split(split),
            },
        }

    def _evaluate_rows(self, loaded: LoadedModel, rows: list[dict], threshold: float) -> dict:
        class_to_idx = self.trainer._class_to_idx(rows)
        transform = self.trainer._build_transform(loaded, train=False)
        loader = DataLoader(
            SimpleImageDataset(rows, transform, class_to_idx),
            batch_size=32,
            shuffle=False,
        )
        loaded.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        y_true: list[int] = []
        y_prob: list[float] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.loader.device)
                yb = yb.to(self.loader.device).float().view(-1)
                logits = loaded.model(xb).view(-1)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits).view(-1)
                total_loss += loss.item() * xb.size(0)
                y_true.extend(yb.long().cpu().tolist())
                y_prob.extend(probs.cpu().tolist())
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        metrics["loss"] = total_loss / max(len(y_true), 1)
        return metrics

    def _summary_row(
        self,
        method: str,
        metrics: dict,
        round_number: int | None = None,
        hospital_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "round_number": round_number,
            "hospital_id": hospital_id,
            "loss": metrics.get("loss"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "false_negatives": metrics.get("false_negatives"),
            "false_positives": metrics.get("false_positives"),
            "threshold": metrics.get("threshold"),
            "security_mode": metrics.get("security_mode"),
            "number_of_clients": metrics.get("number_of_clients"),
            "number_of_completed_clients": metrics.get("number_of_completed_clients"),
            "dropped_clients": metrics.get("dropped_clients"),
            "aggregation_time_seconds": metrics.get("aggregation_time_seconds"),
            "communication_size_bytes": metrics.get("communication_size_bytes"),
        }

    def _persist_metrics(
        self,
        run_id: str,
        scope: str,
        metrics: dict,
        round_number: int | None = None,
        hospital_id: str | None = None,
        split: str | None = None,
    ) -> None:
        self.db.save_evaluation_metrics(
            run_id=run_id,
            model_version_id=None,
            scope=scope,
            metrics=metrics,
            round_number=round_number,
            hospital_id=hospital_id,
            split=split,
        )
        self.db.save_confusion_matrix(
            run_id=run_id,
            model_version_id=None,
            scope=scope,
            metrics=metrics,
            round_number=round_number,
            hospital_id=hospital_id,
            split=split,
        )

    def _persist_split_distribution(
        self,
        dataset_id: int,
        run_id: str,
        strategy: str,
        split: dict[str, list[dict]],
        split_name: str,
    ) -> None:
        for hospital_id, hospital_rows in split.items():
            normal = sum(1 for row in hospital_rows if str(row.get("label")).upper() == "NORMAL")
            pneumonia = sum(1 for row in hospital_rows if str(row.get("label")).upper() == "PNEUMONIA")
            nonzero = [count for count in [normal, pneumonia] if count > 0]
            ratio = (max(nonzero) / max(min(nonzero), 1)) if nonzero else None
            self.db.save_dataset_distribution(
                dataset_id=dataset_id,
                run_id=run_id,
                split_strategy=strategy,
                hospital_id=hospital_id,
                split=split_name,
                total_count=len(hospital_rows),
                normal_count=normal,
                pneumonia_count=pneumonia,
                imbalance_ratio=ratio,
                details={"source": "non_iid_simulation"},
            )
