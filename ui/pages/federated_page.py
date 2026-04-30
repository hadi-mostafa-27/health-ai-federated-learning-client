from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.dataset_manager import DatasetManager
from core.db import DatabaseManager
from core.fl_client import FLClient
from ui.pages.base import BasePage


class FederatedPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__(
            "Federated Aggregation Prototype",
            "Exchange model checkpoints and metadata. Secure aggregation and differential privacy are not implemented.",
        )
        self.config = config
        self.db = db
        self.dataset_manager = DatasetManager(db)
        self.client = FLClient(config.get("server_url"), config.get("api_token"), db)

        btn_widget = QWidget()
        layout = QVBoxLayout(btn_widget)
        self.ping_btn = QPushButton("Ping Server")
        self.round_btn = QPushButton("Check Current Round")
        self.fetch_btn = QPushButton("Fetch Global Model")
        self.send_btn = QPushButton("Send Local Model Update")
        for btn in [self.ping_btn, self.round_btn, self.fetch_btn, self.send_btn]:
            layout.addWidget(btn)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(btn_widget)
        self.layout.addWidget(self.output)

        self.ping_btn.clicked.connect(self.ping)
        self.round_btn.clicked.connect(self.current_round)
        self.fetch_btn.clicked.connect(self.fetch_model)
        self.send_btn.clicked.connect(self.send_update)

    def ping(self) -> None:
        try:
            self.output.setPlainText(json.dumps(self.client.ping(), indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "FL Error", str(exc))

    def current_round(self) -> None:
        try:
            payload = self.client.current_round()
            self.output.setPlainText(json.dumps(payload, indent=2))
            self.db.execute(
                """
                INSERT INTO fl_rounds (
                    round_number, global_model_version, status, aggregation_algorithm,
                    global_metrics_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload.get("round", 0),
                    payload.get("version", "unknown"),
                    "checked",
                    payload.get("aggregation_algorithm"),
                    json.dumps(payload),
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "FL Error", str(exc))

    def fetch_model(self) -> None:
        try:
            payload = self.client.fetch_latest_model(Path(self.config.get("models_dir")) / "server_downloads")
            self.output.setPlainText(json.dumps(payload, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "FL Error", str(exc))

    def send_update(self) -> None:
        try:
            local_dir = Path(self.config.get("models_dir")) / "local"
            checkpoint = local_dir / "local_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Expected checkpoint at {checkpoint}")
            dataset = self.dataset_manager.latest_dataset()
            num_samples = int(dataset["train_count"]) if dataset else 0
            metadata = {
                "hospital_id": self.config.get("hospital_id"),
                "num_samples": num_samples,
                "note": "Prototype update. Model updates may leak information; secure aggregation and DP are not implemented.",
            }
            payload = self.client.send_model_update(self.config.get("hospital_id"), checkpoint, metadata=metadata)
            self.output.setPlainText(json.dumps(payload, indent=2))
            self.db.save_client_update(
                hospital_id=self.config.get("hospital_id"),
                num_samples=num_samples,
                local_loss=None,
                local_accuracy=None,
                local_metrics=metadata,
                status="sent",
                update_path=str(checkpoint),
            )
        except Exception as exc:
            QMessageBox.critical(self, "FL Error", str(exc))
