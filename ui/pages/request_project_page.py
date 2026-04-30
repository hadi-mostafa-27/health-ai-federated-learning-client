import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, 
    QTextEdit, QPushButton, QFormLayout, QFrame, QMessageBox,
    QScrollArea, QWidget, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox
)
from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage
from ui.pages.project_runner_page import CollapsibleBox

class RequestProjectPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Request New AI Collaboration")
        self.config = config
        self.db = db
        self.logged_hospital_id = self.config.get("hospital_id", "HOSP_DEMO")
        self.logged_hospital_name = self.config.get("hospital_name", "Demo Hospital")
        
        self.layout.setSpacing(18)
        
        title = QLabel("Request New AI Collaboration")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")
        
        subtitle = QLabel("Propose a full FL project configuration for Admin approval.")
        subtitle.setStyleSheet("font-size:13px; color:#64748B;")
        
        self.layout.addWidget(title)
        self.layout.addWidget(subtitle)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0,0,0,0)
        c_layout.setSpacing(18)
        
        card = QFrame()
        card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 20, 20, 20)
        
        form = QFormLayout()
        
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("e.g. Multi-center Pneumonia Detection")
        
        self.disease_target = QComboBox()
        self.disease_target.addItems(["Pneumonia", "Pleural Effusion", "Tuberculosis", "Atelectasis", "Custom"])
        
        self.task_type = QComboBox()
        self.task_type.addItems(["Image Classification", "CSV/tabular classification"])
        
        self.dataset_type = QComboBox()
        self.dataset_type.addItems(["X-ray image folders", "CSV tabular format"])
        
        self.suggested_backbone = QComboBox()
        self.suggested_backbone.addItems(["DenseNet121", "ResNet18", "ResNet50", "Custom"])
        
        self.fl_algorithm = QComboBox()
        self.fl_algorithm.addItems(["FedAvg", "FedProx"])
        
        self.reason = QTextEdit()
        self.reason.setPlaceholderText("Explain the clinical motivation and why other hospitals should join...")
        self.reason.setMaximumHeight(80)
        
        form.addRow("Project Name", self.project_name)
        form.addRow("Disease Target", self.disease_target)
        form.addRow("Task Type", self.task_type)
        form.addRow("Dataset Schema", self.dataset_type)
        form.addRow("Suggested Model Backbone", self.suggested_backbone)
        form.addRow("FL Algorithm", self.fl_algorithm)
        form.addRow("Clinical Motivation", self.reason)
        
        outer.addLayout(form)
        
        # Advanced Settings
        advanced = CollapsibleBox("Advanced Training Settings")
        adv = QFormLayout()
        
        self.total_rounds = QSpinBox()
        self.total_rounds.setRange(1, 100)
        self.total_rounds.setValue(5)
        
        self.local_epochs = QSpinBox()
        self.local_epochs.setRange(1, 50)
        self.local_epochs.setValue(1)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(8)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setValue(0.0001)
        
        self.participation_fraction = QDoubleSpinBox()
        self.participation_fraction.setDecimals(2)
        self.participation_fraction.setRange(0.1, 1.0)
        self.participation_fraction.setValue(1.0)
        
        self.stop_accuracy = QDoubleSpinBox()
        self.stop_accuracy.setDecimals(4)
        self.stop_accuracy.setRange(0.0, 1.0)
        self.stop_accuracy.setValue(0.92)
        
        adv.addRow("Total Rounds", self.total_rounds)
        adv.addRow("Local Epochs", self.local_epochs)
        adv.addRow("Batch Size", self.batch_size)
        adv.addRow("Learning Rate", self.learning_rate)
        adv.addRow("Participation Fraction", self.participation_fraction)
        adv.addRow("Target / Stop Accuracy", self.stop_accuracy)
        
        advanced.setContentLayout(adv)
        outer.addWidget(advanced)
        
        # Requested Hospitals
        hospitals_box = QGroupBox("Requested Participating Hospitals")
        hospitals_layout = QVBoxLayout(hospitals_box)
        
        self.hospital_checks = {}
        active_hospitals = self.db.list_hospitals(active_only=True)
        if not active_hospitals:
            hospitals_layout.addWidget(QLabel("No hospitals found."))
        else:
            for h in active_hospitals:
                cb = QCheckBox(f"{h['hospital_name']} ({h['hospital_id']})")
                if h['hospital_id'] == self.logged_hospital_id:
                    cb.setChecked(True)
                    cb.setEnabled(False)
                self.hospital_checks[h['hospital_id']] = (cb, h['hospital_name'])
                hospitals_layout.addWidget(cb)
        
        outer.addWidget(hospitals_box)
        
        self.send_btn = QPushButton("Send Full Project Request to Admin")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #0D2D4F;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #1A3F5F;
            }
        """)
        self.send_btn.clicked.connect(self.submit_request)
        outer.addWidget(self.send_btn)
        
        c_layout.addWidget(card)
        c_layout.addStretch()
        
        scroll.setWidget(container)
        self.layout.addWidget(scroll)

    def submit_request(self):
        name = self.project_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Project name is required.")
            return

        profile = self.db.get_hospital(self.logged_hospital_id)
        if str(profile["node_status"] if profile else "inactive").lower() != "active":
            QMessageBox.warning(
                self,
                "Inactive Hospital",
                "Inactive / unavailable hospitals cannot request new FL participation until reactivated.",
            )
            return
            
        requested_hospitals = []
        for hid, (cb, hname) in self.hospital_checks.items():
            if cb.isChecked():
                requested_hospitals.append((hid, hname))
                
        details = {
            "task_type": self.task_type.currentText(),
            "total_rounds": self.total_rounds.value(),
            "local_epochs": self.local_epochs.value(),
            "batch_size": self.batch_size.value(),
            "learning_rate": self.learning_rate.value(),
            "participation_fraction": self.participation_fraction.value(),
            "stop_accuracy": self.stop_accuracy.value(),
            "requested_hospitals": requested_hospitals,
            "requesting_hospital_id": self.logged_hospital_id
        }
        
        self.db.create_project_request(
            hospital_name=self.logged_hospital_name,
            project_name=name,
            disease_target=self.disease_target.currentText(),
            dataset_type=self.dataset_type.currentText(),
            suggested_backbone=self.suggested_backbone.currentText(),
            fl_algorithm=self.fl_algorithm.currentText(),
            reason=self.reason.toPlainText(),
            details_json=json.dumps(details)
        )
        
        self.db.log("project_request", f"{self.logged_hospital_name} requested project: {name}", "pending")
        QMessageBox.information(self, "Success", "Your complete project request has been sent to the Admin.")
        
        self.project_name.clear()
        self.reason.clear()
