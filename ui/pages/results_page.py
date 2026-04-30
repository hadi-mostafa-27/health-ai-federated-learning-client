from __future__ import annotations

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from core.docker_exporter import DockerExportError, DockerPackageExporter
from core.report_generator import ReportGenerator
from ui.pages.base import BasePage


class ResultsPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__("Results", "Predictions, evaluation metrics, and clinically important error counts.")
        self.config = config
        self.db = db
        self.reporter = ReportGenerator(db)
        self.docker_exporter = DockerPackageExporter(db)

        header = QFrame()
        header.setObjectName("PanelCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 18)
        header_layout.setSpacing(14)

        title = QLabel("Prediction and Evaluation History")
        title.setObjectName("SectionTitle")
        header_layout.addWidget(title)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumHeight(40)
        refresh_btn.clicked.connect(self.refresh)

        export_btn = QPushButton("Export Report")
        export_btn.setMinimumHeight(40)
        export_btn.setObjectName("PrimaryButton")
        export_btn.clicked.connect(self.export_report)

        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        header_layout.addWidget(export_btn)
        self.layout.addWidget(header)

        self.prediction_table = self._table(["ID", "Image", "Prediction", "Confidence", "Time (s)", "Date"])
        self.metric_table = self._table([
            "Scope", "Round", "Hospital", "Split", "Accuracy", "F1", "ROC-AUC",
            "Sensitivity", "Specificity", "False Negatives", "False Positives", "Threshold",
        ])
        self.model_table = self._table(["Version", "Architecture", "Source", "Algorithm", "Threshold", "Created"])
        self.cm_table = self._table(["Scope", "Round", "Hospital", "Split", "TN", "FP", "FN", "TP"])
        self.distribution_table = self._table(["Strategy", "Hospital", "Split", "Total", "Normal", "Pneumonia", "Imbalance"])
        self.docker_table = self._table([
            "Project ID", "Project", "Hospital ID", "Hospital", "Status",
            "Rounds", "Final Accuracy", "Final Loss", "Docker Export",
        ])

        tabs = QTabWidget()
        tabs.setObjectName("ResultsTabs")
        tabs.setDocumentMode(True)
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(self._panel("Prediction History", self.prediction_table), "Predictions")
        tabs.addTab(self._panel("Dataset Distributions", self.distribution_table), "Dataset Distributions")
        tabs.addTab(self._panel("Evaluation Metrics", self.metric_table), "Evaluation Metrics")
        tabs.addTab(self._panel("Confusion Matrices", self.cm_table), "Confusion Matrices")
        tabs.addTab(self._panel("Model Version History", self.model_table), "Model Versions")
        tabs.addTab(self._docker_panel(), "Docker Packages")
        self.layout.addWidget(tabs, 1)
        self.refresh()

    def _table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setWordWrap(False)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setMinimumHeight(560)
        return table

    def _panel(self, title: str, table: QTableWidget) -> QFrame:
        panel = QFrame()
        panel.setObjectName("PanelCard")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        layout.addWidget(table, 1)
        return panel

    def _docker_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("PanelCard")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        label = QLabel("Completed Project Docker Packages")
        label.setObjectName("SectionTitle")
        hint = QLabel("Export is enabled only for completed FL projects and joined hospitals.")
        hint.setObjectName("CardMeta")
        self.export_docker_btn = QPushButton("Export Docker Package")
        self.export_docker_btn.setObjectName("PrimaryButton")
        self.export_docker_btn.clicked.connect(self.export_selected_docker_package)
        layout.addWidget(label)
        layout.addWidget(hint)
        layout.addWidget(self.docker_table, 1)
        layout.addWidget(self.export_docker_btn)
        return panel

    def refresh(self) -> None:
        self._refresh_predictions()
        self._refresh_distributions()
        self._refresh_metrics()
        self._refresh_confusion_matrices()
        self._refresh_models()
        self._refresh_docker_exports()

    def _refresh_predictions(self) -> None:
        rows = self.db.fetchall("SELECT * FROM predictions ORDER BY id DESC LIMIT 100")
        self.prediction_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            image_path = str(row["image_path"])
            short_path = image_path.split("/")[-1] if "/" in image_path else image_path.split("\\")[-1]
            values = [
                row["id"],
                short_path,
                row["predicted_label"],
                f"{float(row['confidence']):.2%}",
                f"{float(row['inference_time']):.3f}",
                str(row["created_at"]).split(" ")[0] if row["created_at"] else "N/A",
            ]
            for col, value in enumerate(values):
                self.prediction_table.setItem(i, col, QTableWidgetItem(str(value)))
        self.prediction_table.resizeColumnsToContents()

    def _refresh_distributions(self) -> None:
        rows = self.db.fetchall("SELECT * FROM dataset_distributions ORDER BY id DESC LIMIT 100")
        self.distribution_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row["split_strategy"],
                row["hospital_id"] or "-",
                row["split"],
                row["total_count"],
                row["normal_count"],
                row["pneumonia_count"],
                self._fmt(row["imbalance_ratio"]),
            ]
            for col, value in enumerate(values):
                self.distribution_table.setItem(i, col, QTableWidgetItem(str(value)))
        self.distribution_table.resizeColumnsToContents()

    def _refresh_metrics(self) -> None:
        rows = self.db.fetchall("SELECT * FROM evaluation_metrics ORDER BY id DESC LIMIT 100")
        self.metric_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row["scope"],
                row["round_number"],
                row["hospital_id"] or "-",
                row["split"] or "-",
                self._fmt(row["accuracy"]),
                self._fmt(row["f1_score"]),
                self._fmt(row["roc_auc"]),
                self._fmt(row["sensitivity"]),
                self._fmt(row["specificity"]),
                row["false_negatives"],
                row["false_positives"],
                self._fmt(row["threshold"]),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 9 and row["false_negatives"] and int(row["false_negatives"]) > 0:
                    item.setToolTip("False negatives are clinically serious for pneumonia screening.")
                    item.setBackground(QBrush(QColor("#fee2e2")))
                    item.setForeground(QBrush(QColor("#991b1b")))
                self.metric_table.setItem(i, col, item)
        self.metric_table.resizeColumnsToContents()

    def _refresh_confusion_matrices(self) -> None:
        rows = self.db.fetchall("SELECT * FROM confusion_matrices ORDER BY id DESC LIMIT 100")
        self.cm_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row["scope"],
                row["round_number"],
                row["hospital_id"] or "-",
                row["split"] or "-",
                row["true_negative"],
                row["false_positive"],
                row["false_negative"],
                row["true_positive"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 6 and value and int(value) > 0:
                    item.setToolTip("False negatives are clinically serious for pneumonia screening.")
                    item.setBackground(QBrush(QColor("#fee2e2")))
                    item.setForeground(QBrush(QColor("#991b1b")))
                self.cm_table.setItem(i, col, item)
        self.cm_table.resizeColumnsToContents()

    def _refresh_models(self) -> None:
        rows = self.db.fetchall("SELECT * FROM model_versions ORDER BY id DESC LIMIT 100")
        self.model_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row["version"],
                row["architecture"],
                row["source"],
                row["aggregation_algorithm"] or "-",
                self._fmt(row["threshold"]),
                row["created_at"],
            ]
            for col, value in enumerate(values):
                self.model_table.setItem(i, col, QTableWidgetItem(str(value)))
        self.model_table.resizeColumnsToContents()

    def _refresh_docker_exports(self) -> None:
        role = self.config.get("user_role", "hospital")
        hospital_id = self.config.get("hospital_id", "")
        if role == "admin":
            rows = self.db.fetchall(
                """
                SELECT p.*, pm.hospital_id, pm.hospital_name, pm.status AS membership_status
                FROM fl_projects p
                JOIN project_memberships pm ON pm.project_id = p.id
                WHERE p.status = 'completed' AND pm.status LIKE '%joined%'
                ORDER BY p.completed_at DESC, p.id DESC
                """
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT p.*, pm.hospital_id, pm.hospital_name, pm.status AS membership_status
                FROM fl_projects p
                JOIN project_memberships pm ON pm.project_id = p.id
                WHERE p.status = 'completed'
                  AND pm.status LIKE '%joined%'
                  AND pm.hospital_id = ?
                ORDER BY p.completed_at DESC, p.id DESC
                """,
                (hospital_id,),
            )

        self.docker_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            metrics = self.db.get_latest_project_metrics(int(row["id"]))
            export = self.db.fetchone(
                """
                SELECT zip_path FROM docker_exports
                WHERE project_id = ? AND hospital_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (row["id"], row["hospital_id"]),
            )
            values = [
                row["id"],
                row["project_name"],
                row["hospital_id"],
                row["hospital_name"],
                row["status"],
                row["total_rounds"],
                self._fmt(metrics.get("accuracy", metrics.get("acc"))),
                self._fmt(metrics.get("loss")),
                export["zip_path"] if export else "Available",
            ]
            for col, value in enumerate(values):
                self.docker_table.setItem(i, col, QTableWidgetItem(str(value)))
        self.docker_table.resizeColumnsToContents()
        self.export_docker_btn.setEnabled(len(rows) > 0)

    def _fmt(self, value) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    def export_report(self) -> None:
        try:
            pred_rows = self.db.fetchall("SELECT * FROM predictions ORDER BY id DESC LIMIT 100")
            metric_rows = self.db.fetchall("SELECT * FROM evaluation_metrics ORDER BY id DESC LIMIT 100")
            cm_rows = self.db.fetchall("SELECT * FROM confusion_matrices ORDER BY id DESC LIMIT 100")
            payload = {
                "predictions": [dict(r) for r in pred_rows],
                "evaluation_metrics": [dict(r) for r in metric_rows],
                "confusion_matrices": [dict(r) for r in cm_rows],
            }
            path = self.reporter.save_results_report(payload)
            QMessageBox.information(self, "Export Success", f"Report exported to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to export report:\n{str(exc)}")

    def export_selected_docker_package(self) -> None:
        row = self.docker_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Docker Export", "Select a completed project row first.")
            return
        project_id = int(self.docker_table.item(row, 0).text())
        hospital_id = self.docker_table.item(row, 2).text()
        try:
            export = self.docker_exporter.export_for_hospital(
                project_id=project_id,
                hospital_id=hospital_id,
                requester_role=self.config.get("user_role", "hospital"),
            )
            QMessageBox.information(
                self,
                "Docker Export Complete",
                f"Docker package created:\n{export['zip_path']}",
            )
            self._refresh_docker_exports()
        except DockerExportError as exc:
            QMessageBox.warning(self, "Docker Export", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Docker Export Error", f"Failed to export Docker package:\n{exc}")
