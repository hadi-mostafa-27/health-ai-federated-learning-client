from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLabel,
)

from core.config_manager import ConfigManager
from core.dataset_manager import DatasetManager
from core.db import DatabaseManager
from core.model_loader import NotebookAwareModelLoader
from core.trainer import LocalTrainer, TrainingConfig
from ui.pages.base import BasePage


class TrainingPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Local Training", "Fine-tune DenseNet121 with validation metrics and threshold tuning.")
        self.config = config
        self.db = db
        self.loader = NotebookAwareModelLoader(config.get("models_dir"), config.get("device", "auto"))
        self.dataset_manager = DatasetManager(db)
        self.trainer = LocalTrainer(db, self.loader)
        self.thread = None

        config_frame = QFrame()
        config_frame.setObjectName("PanelCard")
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(24, 20, 24, 20)
        config_layout.setSpacing(16)

        config_title = QLabel("Training Configuration")
        config_title.setObjectName("SectionTitle")
        config_layout.addWidget(config_title)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setSpacing(14)
        form.setContentsMargins(0, 0, 0, 0)

        self.algorithm = QComboBox()
        self.algorithm.addItems(["FedAvg", "FedProx"])

        self.epochs = QSpinBox()
        self.epochs.setValue(1)
        self.epochs.setMaximum(100)
        self.epochs.setMinimum(1)

        self.batch = QSpinBox()
        self.batch.setValue(8)
        self.batch.setMaximum(256)
        self.batch.setMinimum(1)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setValue(0.0001)

        self.form_mu = QDoubleSpinBox()
        self.form_mu.setDecimals(4)
        self.form_mu.setRange(0.0, 10.0)
        self.form_mu.setValue(0.01)

        self.threshold_strategy = QComboBox()
        self.threshold_strategy.addItems(["best_f1", "high_sensitivity", "balanced", "fixed_0_5"])

        self.class_weighting = QCheckBox("Use BCE positive-class weighting")
        self.class_weighting.setChecked(True)
        self.weighted_sampler = QCheckBox("Use weighted sampler")

        form.addRow("Algorithm", self.algorithm)
        form.addRow("Epochs", self.epochs)
        form.addRow("Batch Size", self.batch)
        form.addRow("Learning Rate", self.learning_rate)
        form.addRow("FedProx mu", self.form_mu)
        form.addRow("Threshold Strategy", self.threshold_strategy)
        form.addRow("Class Weighting", self.class_weighting)
        form.addRow("Weighted Sampler", self.weighted_sampler)

        config_layout.addWidget(form_widget)
        self.layout.addWidget(config_frame)

        actions_frame = QFrame()
        actions_frame.setObjectName("PanelCard")
        actions_layout = QHBoxLayout(actions_frame)
        actions_layout.setContentsMargins(24, 18, 24, 18)
        actions_layout.setSpacing(14)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setMinimumHeight(44)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(44)
        actions_layout.addWidget(self.start_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addStretch()
        self.layout.addWidget(actions_frame)

        log_frame = QFrame()
        log_frame.setObjectName("PanelCard")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(24, 20, 24, 20)
        log_layout.setSpacing(12)

        log_title = QLabel("Training Progress")
        log_title.setObjectName("SectionTitle")
        log_layout.addWidget(log_title)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Training logs will appear here...")
        log_layout.addWidget(self.log)

        self.layout.addWidget(log_frame, 1)

        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)

    def start_training(self) -> None:
        try:
            loaded = self.loader.load_default()
            dataset = self.dataset_manager.latest_dataset()
            if not dataset:
                raise RuntimeError("Register a dataset first.")
            train_rows = self.dataset_manager.images_for_split(int(dataset["id"]), "train")
            val_rows = self.dataset_manager.images_for_split(int(dataset["id"]), "val")
            algorithm = self.algorithm.currentText()
            cfg = TrainingConfig(
                epochs=int(self.epochs.value()),
                batch_size=int(self.batch.value()),
                learning_rate=float(self.learning_rate.value()),
                aggregation_algorithm=algorithm,
                fedprox_mu=float(self.form_mu.value()) if algorithm == "FedProx" else 0.0,
                threshold_strategy=self.threshold_strategy.currentText(),
                class_weighting=self.class_weighting.isChecked(),
                weighted_sampler=self.weighted_sampler.isChecked(),
            )
            self.log.clear()
            self.log.append(f"Training started. Algorithm={cfg.aggregation_algorithm}, FedProx mu={cfg.fedprox_mu}")
            self.thread = self.trainer.run_async(
                loaded=loaded,
                train_rows=train_rows,
                val_rows=val_rows,
                config=cfg,
                checkpoint_dir=Path(self.config.get("models_dir")) / "local",
                on_progress=self.on_progress,
                on_done=self.on_done,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Training Error", str(exc))

    def stop_training(self) -> None:
        self.trainer.stop()
        self.log.append("Stop requested.")

    def on_progress(self, row: dict) -> None:
        self.log.append(
            "Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            "val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, f1={val_f1_score:.4f}, "
            "sens={val_sensitivity:.4f}, spec={val_specificity:.4f}, FN={val_false_negatives}, "
            "FP={val_false_positives}, threshold={threshold:.3f}".format(**row)
        )

    def on_done(self, result: dict) -> None:
        metrics = result.get("best_metrics", {})
        self.log.append(
            "Done. Best accuracy={:.4f}, F1={:.4f}, ROC-AUC={}, threshold={:.3f}, checkpoint={}".format(
                result.get("best_accuracy", 0.0),
                metrics.get("f1_score", 0.0),
                metrics.get("roc_auc"),
                result.get("threshold", 0.5),
                result.get("best_path"),
            )
        )
