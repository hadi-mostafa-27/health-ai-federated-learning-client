from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLineEdit, QMessageBox, QPushButton, QWidget

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage


class SettingsPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Settings")
        self.config = config
        self.db = db
        widget = QWidget()
        form = QFormLayout(widget)
        self.fields = {}
        for key in ["hospital_id", "hospital_name", "country", "city", "models_dir", "dataset_dir", "reports_dir"]:
            edit = QLineEdit(str(config.get(key, "")))
            self.fields[key] = edit
            form.addRow(key, edit)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save)
        form.addRow(save_btn)
        self.layout.addWidget(widget)

    def save(self) -> None:
        self.config.update({k: v.text().strip() for k, v in self.fields.items()})
        QMessageBox.information(self, "Settings", "Saved")
