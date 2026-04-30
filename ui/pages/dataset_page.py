from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from core.config_manager import ConfigManager
from core.data_generator import generate_sample_dataset
from core.dataset_manager import DatasetManager
from core.db import DatabaseManager
from ui.pages.base import BasePage


class DatasetPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Dataset Management", "Validate NORMAL/PNEUMONIA folders and create reproducible splits.")
        self.config = config
        self.db = db
        self.manager = DatasetManager(db)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.name_edit = QLineEdit("Local Chest X-ray Dataset")
        self.path_edit = QLineEdit(config.get("dataset_dir"))
        self.csv_edit = QLineEdit("")

        self.train_ratio = self._ratio_spin(0.70)
        self.val_ratio = self._ratio_spin(0.15)
        self.test_ratio = self._ratio_spin(0.15)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)

        btn_row = QWidget()
        row = QHBoxLayout(btn_row)
        browse_btn = QPushButton("Browse Folder")
        browse_csv = QPushButton("Browse CSV")
        demo_btn = QPushButton("Generate Demo Data")
        register_btn = QPushButton("Register Dataset")
        row.addWidget(browse_btn)
        row.addWidget(browse_csv)
        row.addWidget(demo_btn)
        row.addWidget(register_btn)

        form.addRow("Dataset Name", self.name_edit)
        form.addRow("Folder", self.path_edit)
        form.addRow("CSV (optional)", self.csv_edit)
        form.addRow("Train Ratio", self.train_ratio)
        form.addRow("Validation Ratio", self.val_ratio)
        form.addRow("Test Ratio", self.test_ratio)
        form.addRow("Random Seed", self.seed)
        form.addRow(btn_row)
        self.layout.addWidget(form_widget)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        browse_btn.clicked.connect(self.browse_folder)
        browse_csv.clicked.connect(self.browse_csv)
        demo_btn.clicked.connect(self.generate_demo_data)
        register_btn.clicked.connect(self.register)

    def _ratio_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.05, 0.95)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def browse_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", str(Path.cwd()), "CSV Files (*.csv)")
        if file_path:
            self.csv_edit.setText(file_path)

    def register(self) -> None:
        try:
            summary = self.manager.register_dataset(
                dataset_name=self.name_edit.text().strip() or "Dataset",
                dataset_path=self.path_edit.text().strip(),
                label_csv_path=self.csv_edit.text().strip() or None,
                train_ratio=float(self.train_ratio.value()),
                val_ratio=float(self.val_ratio.value()),
                test_ratio=float(self.test_ratio.value()),
                random_seed=int(self.seed.value()),
            )
            warnings = "\n".join(f"- {w}" for w in (summary.warnings or [])) or "None"
            self.output.setPlainText(
                "Registered dataset:\n"
                f"samples={summary.num_samples}\n"
                f"classes={summary.num_classes}\n"
                f"class_distribution={summary.class_distribution}\n"
                f"train={summary.train_count}, val={summary.val_count}, test={summary.test_count}\n"
                f"split_distribution={summary.split_distribution}\n"
                f"imbalance_ratio={summary.imbalance_ratio}\n"
                f"invalid_images={len(summary.invalid_images or [])}\n"
                f"warnings:\n{warnings}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Dataset Error", str(exc))

    def generate_demo_data(self) -> None:
        try:
            self.output.setPlainText("Generating demo dataset (simulated X-rays)...")
            demo_path = Path.cwd() / "sample_data" / "chest_xray"
            generate_sample_dataset(demo_path)
            self.path_edit.setText(str(demo_path))
            self.name_edit.setText("Demo Synthetic Dataset")
            self.csv_edit.setText("")
            self.output.setPlainText(f"Generated demo dataset at {demo_path}.\nNow registering...")
            self.register()
        except Exception as exc:
            QMessageBox.critical(self, "Demo Generation Error", str(exc))
