from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QGridLayout, QLabel, QMessageBox, QPushButton, QTextEdit, QWidget

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from core.gradcam_engine import GradCAMEngine
from core.model_loader import NotebookAwareModelLoader
from ui.pages.base import BasePage


class GradCAMPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Grad-CAM", "Visual explanation aid; not clinical proof.")
        self.config = config
        self.db = db
        self.loader = NotebookAwareModelLoader(config.get("models_dir"), config.get("device", "auto"))
        self.engine = GradCAMEngine()
        self.current_image: str | None = None

        w = QWidget()
        grid = QGridLayout(w)
        self.orig = QLabel("Original")
        self.overlay = QLabel("Overlay")
        self.orig.setMinimumHeight(300)
        self.overlay.setMinimumHeight(300)
        choose_btn = QPushButton("Choose Image")
        gen_btn = QPushButton("Generate Grad-CAM")
        self.disclaimer = QTextEdit()
        self.disclaimer.setReadOnly(True)
        self.disclaimer.setMaximumHeight(90)
        self.disclaimer.setPlainText(GradCAMEngine.DISCLAIMER)

        grid.addWidget(self.orig, 0, 0)
        grid.addWidget(self.overlay, 0, 1)
        grid.addWidget(choose_btn, 1, 0)
        grid.addWidget(gen_btn, 1, 1)
        grid.addWidget(self.disclaimer, 2, 0, 1, 2)
        self.layout.addWidget(w)

        choose_btn.clicked.connect(self.choose)
        gen_btn.clicked.connect(self.generate)

    def choose(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", str(Path.cwd()), "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            self.current_image = file_path
            self.orig.setPixmap(QPixmap(file_path).scaledToHeight(280))

    def generate(self) -> None:
        if not self.current_image:
            QMessageBox.warning(self, "Grad-CAM", "Load an image first")
            return
        try:
            loaded = self.loader.load_default()
            result = self.engine.generate_overlay(self.current_image, self.config.get("visualizations_dir"), loaded_model=loaded)
            display_path = result.get("comparison_path") or result["overlay_path"]
            self.overlay.setPixmap(QPixmap(display_path).scaledToHeight(280))
            self.disclaimer.setPlainText(
                f"{result.get('disclaimer', GradCAMEngine.DISCLAIMER)}\n"
                f"Target layer: {result.get('target_layer')}; target class: {result.get('target_class')}; "
                f"predicted class: {result.get('predicted_class')}"
            )
            pred_row = self.db.fetchone("SELECT id FROM predictions ORDER BY id DESC LIMIT 1")
            pred_id = int(pred_row[0]) if pred_row else None
            self.db.execute(
                "INSERT INTO gradcam_outputs (prediction_id, heatmap_path, overlay_path, target_layer) VALUES (?, ?, ?, ?)",
                (pred_id, result["heatmap_path"], result["overlay_path"], result["target_layer"]),
            )
            self.db.log("gradcam", f"Generated Grad-CAM for {Path(self.current_image).name}", "success")
        except Exception as exc:
            QMessageBox.critical(self, "Grad-CAM Error", str(exc))
