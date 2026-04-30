from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QGridLayout, QLabel, QMessageBox, QPushButton,
    QTextEdit, QWidget, QFrame, QVBoxLayout, QHBoxLayout
)

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from core.inference import InferenceEngine
from core.model_loader import NotebookAwareModelLoader
from ui.pages.base import BasePage


class PredictionPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Prediction", "Run model inference on X-ray images using the saved operating threshold.")
        self.config = config
        self.db = db
        self.loader = NotebookAwareModelLoader(config.get("models_dir"), config.get("device", "auto"))
        self.engine = InferenceEngine(self.loader, db)
        self.current_image: str | None = None

        # Image preview and controls
        preview_frame = QFrame()
        preview_frame.setObjectName("PanelCard")
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(24, 20, 24, 20)
        preview_layout.setSpacing(24)

        # Image preview
        preview_container = QWidget()
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container_layout.setSpacing(12)

        preview_label = QLabel("X-Ray Preview")
        preview_label.setObjectName("SectionTitle")
        preview_label.setStyleSheet("font-size: 18px;")
        preview_container_layout.addWidget(preview_label)

        self.preview = QLabel("No image loaded")
        self.preview.setMinimumWidth(320)
        self.preview.setMinimumHeight(320)
        self.preview.setStyleSheet(
            "border: 2px dashed #cbd5e1; border-radius: 12px; padding: 20px; "
            "background: #f8fafc; color: #94a3b8; text-align: center;"
        )
        self.preview.setAlignment(Qt.AlignCenter)
        preview_container_layout.addWidget(self.preview)

        load_btn = QPushButton("📁 Upload X-ray Image")
        load_btn.setMinimumHeight(44)
        preview_container_layout.addWidget(load_btn)

        preview_layout.addWidget(preview_container)

        # Results section
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(12)

        results_label = QLabel("Prediction Results")
        results_label.setObjectName("SectionTitle")
        results_label.setStyleSheet("font-size: 18px;")
        results_layout.addWidget(results_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Results will appear here after running a prediction...")
        results_layout.addWidget(self.output)

        pred_btn = QPushButton("🔍 Run Prediction")
        pred_btn.setObjectName("PrimaryButton")
        pred_btn.setMinimumHeight(44)
        results_layout.addWidget(pred_btn)

        preview_layout.addWidget(results_container)

        self.layout.addWidget(preview_frame)

        # Connect signals
        load_btn.clicked.connect(self.load_image)
        pred_btn.clicked.connect(self.run_prediction)

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select X-ray Image", str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.current_image = file_path
            pix = QPixmap(file_path).scaledToHeight(280, Qt.SmoothTransformation)
            self.preview.setPixmap(pix)
            self.preview.setAlignment(Qt.AlignCenter)

    def run_prediction(self) -> None:
        if not self.current_image:
            QMessageBox.warning(self, "Prediction", "Please load an image first")
            return
        try:
            self.output.setPlainText("⏳ Running prediction...")
            loaded = self.loader.load_default()
            result = self.engine.predict(self.current_image, loaded)
            version = Path(loaded.checkpoint_path).name
            pred_id = self.engine.persist_prediction(self.current_image, result, model_version=version)

            output_text = f"""
🏥 Model Information
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: {version}
Profile: {loaded.notebook_profile}

📊 Prediction Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Diagnosis: {result.predicted_label}
Confidence: {result.confidence:.2%}
Processing Time: {result.elapsed_seconds:.3f}s

📝 Additional Details
━━━━━━━━━━━━━━━━━━━━━━━━━━━
{result.details}
""".strip()
            self.output.setPlainText(output_text)
        except Exception as exc:
            self.output.setPlainText(f"❌ Error: {str(exc)}")
