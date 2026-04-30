from __future__ import annotations

from datetime import datetime
from typing import List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QProgressBar, QScrollArea, QSpinBox, QTextEdit, QToolButton, QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.dataset_manager import DatasetManager
from core.db import DatabaseManager
from core.docker_exporter import DockerExportError, DockerPackageExporter
from core.trainer import LocalTrainer, TrainingConfig
from core.model_loader import NotebookAwareModelLoader
from core.fl_engine import FederatedEngine
from core.non_iid import FederatedSplitConfig, split_federated_rows, summarize_federated_split
from ui.pages.base import BasePage
import random
from pathlib import Path
import threading



class CollapsibleBox(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.toggle_btn = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.clicked.connect(self.on_toggled)

        self.content = QWidget()
        self.content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)

    def on_toggled(self):
        expanded = self.toggle_btn.isChecked()
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)

    def setContentLayout(self, layout):
        self.content.setLayout(layout)


class StepWidget(QFrame):
    def __init__(self, title: str, detail: str):
        super().__init__()
        self.status = QLabel("WAITING")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFixedWidth(110)

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight:700; font-size:14px; color:#0F172A;")

        self.detail = QLabel(detail)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#64748B; font-size:12px;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.status)

        text = QVBoxLayout()
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout.addLayout(text, 1)

        self.set_status("waiting")

    def set_status(self, state: str):
        styles = {
            "waiting": ("WAITING", "#475569", "#E2E8F0"),
            "running": ("RUNNING", "#1D4ED8", "#DBEAFE"),
            "done": ("DONE", "#047857", "#D1FAE5"),
            "error": ("ERROR", "#B91C1C", "#FEE2E2"),
        }
        text, fg, bg = styles[state]
        self.status.setText(text)
        self.status.setStyleSheet(
            f"background:{bg}; color:{fg}; font-weight:700; border-radius:10px; padding:8px;"
        )
        self.setStyleSheet("""
            QFrame {
                background:white;
                border:1px solid #E2E8F0;
                border-radius:14px;
            }
        """)


class HospitalStatusCard(QFrame):
    def __init__(self, hospital_id: str, hospital_name: str):
        super().__init__()
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name

        self.name = QLabel(hospital_name)
        self.name.setStyleSheet("font-weight:700; font-size:13px; color:#0F172A;")

        self.meta = QLabel(hospital_id)
        self.meta.setStyleSheet("color:#64748B; font-size:12px;")

        self.state = QLabel("Not selected")
        self.state.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.name)
        layout.addWidget(self.meta)
        layout.addWidget(self.state)

        self.set_state("Not selected", "idle")
        self.setStyleSheet("""
            QFrame {
                background:white;
                border:1px solid #E2E8F0;
                border-radius:14px;
            }
        """)

    def set_state(self, text: str, kind: str):
        styles = {
            "idle": ("#E2E8F0", "#475569"),
            "selected": ("#DBEAFE", "#1D4ED8"),
            "training": ("#FEF3C7", "#B45309"),
            "sent": ("#E0F2FE", "#0369A1"),
            "done": ("#D1FAE5", "#047857"),
            "success": ("#D1FAE5", "#047857"),
            "waiting": ("#FEF3C7", "#B45309"),
            "unavailable": ("#FEE2E2", "#B91C1C"),
        }
        bg, fg = styles.get(kind, styles["idle"])
        self.state.setText(text)
        self.state.setStyleSheet(
            f"background:{bg}; color:{fg}; font-weight:700; border-radius:10px; padding:6px;"
        )


class ProjectRunnerPage(BasePage):
    training_progress_signal = Signal(dict)
    training_done_signal = Signal(dict)
    eval_done_signal = Signal(dict)

    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Federated Project Setup & Monitor")
        
        self.training_progress_signal.connect(self.on_real_training_progress)
        self.training_done_signal.connect(self.on_real_training_done)
        self.eval_done_signal.connect(self.on_eval_done)

        self.config = config
        self.db = db
        self.fl_engine = None
        self.active_train_threads = 0
        self.current_project_id = None
        self.current_round = 1
        self.current_step = 0
        self.demo_running = False
        self.joined_participants = []
        self.round_participants = []
        self.last_aggregation_record = None
        self.latest_global_metrics = None

        self.user_role = self.config.get("user_role", "admin")
        self.logged_hospital = self.config.get("hospital_name", "Demo Hospital")
        self.logged_hospital_id = self.config.get("hospital_id", self.logged_hospital)

        self.loader = NotebookAwareModelLoader(config.get("models_dir"), config.get("device", "auto"))
        self.dataset_manager = DatasetManager(db)
        self.trainer = LocalTrainer(db, self.loader)
        self.docker_exporter = DockerPackageExporter(db)

        self.layout.setSpacing(18)

        title = QLabel("Federated Project Setup & Monitor")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")

        subtitle = QLabel(
            "Admin creates FL projects, hospitals request or join, and doctors can watch every training step clearly."
        )
        subtitle.setStyleSheet("font-size:13px; color:#64748B;")

        self.layout.addWidget(title)
        self.layout.addWidget(subtitle)
        self.layout.addWidget(self.build_summary_card())
        self.layout.addWidget(self.build_setup_card())
        self.layout.addWidget(self.build_monitor_card())

        self.refresh_project_summary()

    def build_summary_card(self):
        card = QFrame()
        card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        layout = QGridLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)

        self.summary_project = QLabel("No active project")
        self.summary_project.setStyleSheet("font-size:18px; font-weight:800; color:#0F172A;")

        self.summary_owner = QLabel("Created by: -")
        self.summary_approved = QLabel("Approved by: -")
        self.summary_goal = QLabel("Goal: -")
        self.summary_backbone = QLabel("Backbone: -")
        self.summary_algo = QLabel("FL Algorithm: -")
        self.summary_round = QLabel("Current Round: -")
        self.summary_status = QLabel("Status: idle")

        for w in [self.summary_owner, self.summary_approved, self.summary_goal, self.summary_backbone,
                  self.summary_algo, self.summary_round, self.summary_status]:
            w.setStyleSheet("font-size:13px; color:#475569;")

        layout.addWidget(self.summary_project, 0, 0, 1, 3)
        layout.addWidget(self.summary_owner, 1, 0)
        layout.addWidget(self.summary_approved, 1, 1)
        layout.addWidget(self.summary_goal, 1, 2)
        layout.addWidget(self.summary_backbone, 2, 0)
        layout.addWidget(self.summary_algo, 2, 1)
        layout.addWidget(self.summary_status, 2, 2)
        
        # Add a row for current round to keep it neat
        layout.addWidget(self.summary_round, 3, 0)

        self.export_docker_btn = QPushButton("Export Docker Package")
        self.export_docker_btn.setEnabled(False)
        self.export_docker_btn.setToolTip("Available after this FL project is completed.")
        self.export_docker_btn.clicked.connect(self.export_current_project_package)
        layout.addWidget(self.export_docker_btn, 3, 2)
        return card

    def build_setup_card(self):
        card = QFrame()
        card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Project Setup")
        header.setStyleSheet("font-size:18px; font-weight:800; color:#0F172A;")
        outer.addWidget(header)

        role_label = QLabel(f"Logged in as: {self.user_role.upper()} | {self.logged_hospital}")
        role_label.setStyleSheet("color:#2563EB; font-weight:700;")
        outer.addWidget(role_label)

        form = QFormLayout()

        self.project_name = QLineEdit("Multi-center Pneumonia FL Study")

        self.mode_toggle = QComboBox()
        self.mode_toggle.addItems(["Real Federated Training Mode", "Demo Mode (Visual Only)"])
        self.mode_toggle.setStyleSheet("background-color: #FEF3C7; color: #B45309; font-weight: bold;")

        self.created_by = QComboBox()
        if self.user_role == "admin":
            self.created_by.addItems(["Admin"])
        else:
            self.created_by.addItems([f"{self.logged_hospital} Request"])

        self.goal_target = QComboBox()
        self.goal_target.addItems(["Pneumonia", "Pleural Effusion", "Atelectasis", "Tuberculosis", "Custom Goal"])

        self.model_backbone = QComboBox()
        self.model_backbone.addItems(["DenseNet121", "ResNet18", "ResNet50", "Custom Backbone"])

        self.fl_algorithm = QComboBox()
        self.fl_algorithm.addItems(["FedProx", "FedAvg"])

        self.dataset_policy = QComboBox()
        self.dataset_policy.addItems([
            "Each hospital uses its own local dataset",
            "Hospitals use local datasets + server evaluation set",
        ])

        form.addRow("Project Name", self.project_name)
        form.addRow("Operation Mode", self.mode_toggle)
        form.addRow("Created By", self.created_by)
        form.addRow("Goal / Disease Target", self.goal_target)
        form.addRow("Backbone Model", self.model_backbone)
        form.addRow("FL Algorithm", self.fl_algorithm)
        form.addRow("Dataset Policy", self.dataset_policy)
        outer.addLayout(form)

        hospitals_box = QGroupBox("Invited Lebanese Hospitals")
        hospitals_layout = QVBoxLayout(hospitals_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        hospital_widget = QWidget()
        hospital_list_layout = QVBoxLayout(hospital_widget)

        self.hospital_checks = {}
        if self.user_role == "admin":
            all_hospitals = self.db.list_hospitals(active_only=False)
            if not all_hospitals:
                lbl = QLabel("No hospitals found in registry.")
                hospital_list_layout.addWidget(lbl)
            else:
                for h in all_hospitals:
                    is_active = h['node_status'] == 'active'
                    status_text = "" if is_active else " (Inactive / Unavailable)"
                    cb = QCheckBox(f"{h['hospital_name']} ({h['hospital_id']}){status_text}")
                    if not is_active:
                        cb.setEnabled(False)
                        cb.setStyleSheet("color:#B91C1C;")
                    self.hospital_checks[h['hospital_id']] = cb
                    hospital_list_layout.addWidget(cb)
        else:
            cb = QCheckBox(f"{self.config.get('hospital_name', 'My Hospital')} ({self.logged_hospital_id})")
            cb.setChecked(True)
            cb.setEnabled(False)
            if not self._is_hospital_active(self.logged_hospital_id):
                cb.setText(f"{cb.text()} - Inactive / Unavailable")
                cb.setStyleSheet("color:#B91C1C;")
            self.hospital_checks[self.logged_hospital_id] = cb
            hospital_list_layout.addWidget(cb)

        scroll.setWidget(hospital_widget)
        hospitals_layout.addWidget(scroll)
        outer.addWidget(hospitals_box)

        advanced = CollapsibleBox("Advanced Settings")
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

        self.non_iid_strategy = QComboBox()
        self.non_iid_strategy.addItems(["balanced_iid", "label_skew", "quantity_skew"])

        self.imbalance_severity = QDoubleSpinBox()
        self.imbalance_severity.setDecimals(2)
        self.imbalance_severity.setRange(0.0, 0.99)
        self.imbalance_severity.setValue(0.50)

        self.stop_accuracy = QDoubleSpinBox()
        self.stop_accuracy.setDecimals(4)
        self.stop_accuracy.setRange(0.0, 1.0)
        self.stop_accuracy.setValue(0.92)

        adv.addRow("Total Rounds", self.total_rounds)
        adv.addRow("Local Epochs", self.local_epochs)
        adv.addRow("Batch Size", self.batch_size)
        adv.addRow("Learning Rate", self.learning_rate)
        adv.addRow("Participation Fraction", self.participation_fraction)
        adv.addRow("Non-IID Split", self.non_iid_strategy)
        adv.addRow("Imbalance Severity", self.imbalance_severity)
        adv.addRow("Stop Accuracy", self.stop_accuracy)

        advanced.setContentLayout(adv)
        outer.addWidget(advanced)

        btns = QHBoxLayout()

        self.create_btn = QPushButton("Create Project" if self.user_role == "admin" else "Request New FL Project")
        self.start_btn = QPushButton("Start Collaborative Training" if self.user_role == "admin" else "Join Demo Scenario")
        self.reset_btn = QPushButton("Reset Demo")

        self.create_btn.clicked.connect(self.create_project)
        self.start_btn.clicked.connect(self.start_demo)
        self.reset_btn.clicked.connect(self.reset_demo)

        btns.addWidget(self.create_btn)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.reset_btn)
        outer.addLayout(btns)

        return card

    def build_monitor_card(self):
        card = QFrame()
        card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Live Collaboration Monitor")
        title.setStyleSheet("font-size:18px; font-weight:800; color:#0F172A;")
        outer.addWidget(title)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        body = QHBoxLayout()

        left = QVBoxLayout()
        self.step_widgets = [
            StepWidget("Project created", "A collaborative AI project is initialized."),
            StepWidget("Hospitals invited/requested", "Hospitals are selected or request to join."),
            StepWidget("Initial model distributed", "Hospitals receive the same starting model."),
            StepWidget("Local training running", "Each hospital trains on its local dataset."),
            StepWidget("Updates sent", "Only model updates are sent, not patient data."),
            StepWidget("Server aggregation", "The server combines the updates."),
            StepWidget("Global evaluation", "The global model is evaluated."),
            StepWidget("Model redistributed", "The improved model is sent back."),
            StepWidget("Round completed", "One FL round is completed."),
        ]

        for s in self.step_widgets:
            left.addWidget(s)

        right = QVBoxLayout()

        explain_card = QFrame()
        explain_card.setStyleSheet("background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px;")
        explain_layout = QVBoxLayout(explain_card)

        explain_title = QLabel("What is happening now?")
        explain_title.setStyleSheet("font-size:16px; font-weight:800; color:#0F172A;")

        self.explanation_text = QLabel("Create or request a project, then start the scenario.")
        self.explanation_text.setWordWrap(True)
        self.explanation_text.setStyleSheet("font-size:13px; color:#475569;")

        explain_layout.addWidget(explain_title)
        explain_layout.addWidget(self.explanation_text)
        right.addWidget(explain_card)

        self.hospital_cards = []
        self.hospital_cards_layout = QVBoxLayout()
        self.hospital_cards_layout.setContentsMargins(0,0,0,0)
        self.hospital_cards_layout.setSpacing(10)
        
        cards_widget = QWidget()
        cards_widget.setLayout(self.hospital_cards_layout)
        
        scroll = QScrollArea()
        scroll.setWidget(cards_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        right.addWidget(scroll)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        right.addWidget(self.log_view)

        body.addLayout(left, 3)
        body.addLayout(right, 2)
        outer.addLayout(body)
        return card

    def create_project(self):
        participants = self.selected_hospitals()
        if not participants:
            QMessageBox.warning(self, "Project Setup", "Select at least one hospital.")
            return

        import json
        details = {
            "non_iid_strategy": self.non_iid_strategy.currentText(),
            "imbalance_severity": self.imbalance_severity.value(),
            "requested_hospitals": [(hid, self.hospital_checks[hid].text()) for hid in participants],
        }
        project_id = self.db.create_fl_project(
            project_name=self.project_name.text().strip(),
            disease_target=self.goal_target.currentText(),
            model_backbone=self.model_backbone.currentText(),
            fl_algorithm=self.fl_algorithm.currentText(),
            total_rounds=self.total_rounds.value(),
            local_epochs=self.local_epochs.value(),
            batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(),
            participation_fraction=self.participation_fraction.value(),
            stop_accuracy=self.stop_accuracy.value(),
            details_json=json.dumps(details),
        )

        self.current_project_id = int(project_id)
        self.current_round = 1

        if self.user_role == "hospital":
            self.db.log("project_request", f"{self.logged_hospital} requested project #{project_id}", "pending")
            QMessageBox.information(self, "Request Sent", "Project request sent to admin for demo approval.")
            self.append_log(f"{self.logged_hospital} requested a new FL project.")
        else:
            self.db.log("project", f"Admin created project #{project_id}", "success")
            for hospital_id in participants:
                label = self.hospital_checks[hospital_id].text()
                hospital_name = label.split(" (")[0]
                self.db.add_project_membership(project_id, hospital_id, hospital_name, "joined")
            QMessageBox.information(self, "Project Created", "Federated project created successfully.")
            self.append_log("Admin created a new FL project.")

        self.refresh_project_summary()

    def start_demo(self):
        if self.demo_running:
            return

        if not self.current_project_id:
            self.create_project()
            if not self.current_project_id:
                return

        participants = self.selected_hospitals()
        
        # Determine joined hospitals
        joined_participants = []
        if self.current_project_id:
            mems = self.db.list_project_memberships(self.current_project_id)
            all_joined = [m['hospital_id'] for m in mems if 'joined' in m['status']]
            inactive_joined = [hid for hid in all_joined if not self._is_hospital_active(hid)]
            joined_participants = [hid for hid in all_joined if self._is_hospital_active(hid)]
            if not mems:
                joined_participants = participants
            
            # Warn if there are invited hospitals who haven't responded
            invited_count = sum(1 for m in mems if m['status'] == 'invited')
            if invited_count > 0:
                QMessageBox.warning(self, "Pending Invitations", "Some hospitals have not responded yet.")
            if inactive_joined:
                self.append_log(
                    "Inactive / unavailable hospitals skipped this run: "
                    + ", ".join(inactive_joined)
                )
                
            if len(joined_participants) < 2:
                QMessageBox.information(self, "Limited Collaboration", "Only one hospital has joined. Federated collaboration is limited.")
        else:
            joined_participants = participants
        if not joined_participants:
            QMessageBox.warning(self, "No Active Hospitals", "No active joined hospitals are available for this FL run.")
            return
        self.joined_participants = joined_participants
        self.round_participants = list(joined_participants)
            
        self.reset_monitor_only()
        self.demo_running = True
        self.current_step = 0
        self.summary_status.setText("Status: running")
        self.summary_round.setText(f"Current Round: {self.current_round}")

        if "Real" in self.mode_toggle.currentText():
            loaded = self.loader.load_default()
            cfg = TrainingConfig(
                epochs=self.local_epochs.value(),
                batch_size=self.batch_size.value(),
                learning_rate=self.learning_rate.value(),
                aggregation_algorithm=self.fl_algorithm.currentText(),
                fedprox_mu=0.01 if self.fl_algorithm.currentText() == "FedProx" else 0.0,
                participation_fraction=self.participation_fraction.value(),
            )
            self.fl_engine = FederatedEngine(loaded, cfg, self.loader.device)
            dataset_dir = self.config.get("dataset_dir", "data/raw")
            df = self.dataset_manager.scan_folder(dataset_dir)
            if df.empty:
                QMessageBox.warning(self, "No Data", f"No images found in {dataset_dir}! Falling back to Demo Mode.")
                self.mode_toggle.setCurrentIndex(1)
            else:
                rows = df.to_dict('records')
                split = split_federated_rows(
                    rows,
                    FederatedSplitConfig(
                        strategy=self.non_iid_strategy.currentText(),
                        num_hospitals=max(len(joined_participants), 1),
                        seed=cfg.seed,
                        imbalance_severity=self.imbalance_severity.value(),
                    ),
                )
                split_values = list(split.values())
                datasets = {
                    hospital_id: split_values[i] if i < len(split_values) else []
                    for i, hospital_id in enumerate(joined_participants)
                }
                self.fl_engine.prepare_datasets(datasets, self.trainer)
                self.fl_engine.distribute_initial_model(joined_participants)
                self.append_log(
                    f"Initialized PyTorch FL engine with {len(rows)} images, "
                    f"algorithm={self.fl_algorithm.currentText()}, split={self.non_iid_strategy.currentText()}."
                )
                self.append_log(f"Client distribution: {summarize_federated_split(datasets)}")

        for card in self.hospital_cards:
            if not self._is_hospital_active(card.hospital_id):
                card.set_state("Inactive / unavailable", "unavailable")
            elif card.hospital_id in joined_participants:
                card.set_state("Selected for round (Joined)", "selected")
            else:
                card.set_state("Not participating", "idle")

        self.step_sequence = [
            ("Project prepared successfully.", "The AI collaboration project is ready.", 10),
            (f"Hospitals participating: {', '.join(joined_participants)}", "Hospitals are joining this FL scenario.", 20),
            ("Initial model distributed.", "Every selected hospital receives the same starting model.", 32),
            ("Local training started.", "Each hospital trains locally. Image files are not uploaded in this workflow.", 50),
            ("Updates sent to server.", "Only model updates are sent, not images or records.", 66),
            ("Server aggregation running.", "The server combines learning from participating hospitals.", 78),
            ("Global model evaluated.", "The updated model is tested on a reference dataset.", 88),
            ("Improved model redistributed.", "Hospitals receive the improved global model.", 96),
            (f"Round {self.current_round} completed.", "The FL round is complete.", 100),
        ]

        self.run_next_step()

    def run_next_step(self):
        if self.current_step >= len(self.step_sequence):
            if self.current_round < self.total_rounds.value():
                self.current_round += 1
                self.current_step = 0
                self.summary_round.setText(f"Current Round: {self.current_round}")
                self.append_log(f"\n====== Starting FL Round {self.current_round} ======")
                for s in self.step_widgets:
                    s.set_status("waiting")
                self.progress.setValue(0)
            else:
                self.demo_running = False
                self.summary_status.setText("Status: project completed")
                self.finalize_project_completion()
                self.append_log(f"\nAll {self.total_rounds.value()} rounds completed successfully.")
                return

        log, explanation, progress = self.step_sequence[self.current_step]
        if self.current_step == len(self.step_sequence) - 1:
            log = f"Round {self.current_round} completed."

        for i, widget in enumerate(self.step_widgets):
            if i < self.current_step:
                widget.set_status("done")
            elif i == self.current_step:
                widget.set_status("running")
            else:
                widget.set_status("waiting")

        if self.current_step == 2:
            if "Real" in self.mode_toggle.currentText():
                self.fl_engine.client_updates = {}
                self.round_participants = self.fl_engine.select_participants(
                    self.joined_participants,
                    participation_fraction=self.participation_fraction.value(),
                    seed=self.current_round + self.fl_engine.config.seed,
                )
                if not self.round_participants:
                    QMessageBox.critical(self, "FL Error", "No clients selected for this round.")
                    self.demo_running = False
                    return
                self.fl_engine.distribute_initial_model(self.round_participants)
                counts = {h: self.fl_engine.client_sample_counts.get(h, 0) for h in self.round_participants}
                self.append_log(
                    f"Round {self.current_round}: broadcasted global weights to {self.round_participants}. "
                    f"Samples={counts}. Algorithm={self.fl_engine.algorithm}."
                )
            self.current_step += 1
            QTimer.singleShot(1300, self.run_next_step)

        elif self.current_step == 3:
            for card in self.hospital_cards:
                if card.hospital_id in self.round_participants:
                    card.set_state("Training locally", "training")
            
            if "Real" in self.mode_toggle.currentText():
                self.active_train_threads = len(self.round_participants)
                for h in self.round_participants:
                    self.fl_engine.run_local_training_async(
                        hospital_id=h,
                        on_progress=lambda r: self.training_progress_signal.emit(r),
                        on_done=lambda r: self.training_done_signal.emit(r)
                    )
            else:
                self.current_step += 1
                QTimer.singleShot(1300, self.run_next_step)

        elif self.current_step == 4:
            for card in self.hospital_cards:
                if card.hospital_id in self.round_participants:
                    card.set_state("Update sent", "sent")
            self.current_step += 1
            QTimer.singleShot(1300, self.run_next_step)

        elif self.current_step == 5:
            if "Real" in self.mode_toggle.currentText():
                self.append_log(f"Executing sample-weighted {self.fl_engine.algorithm} aggregation.")
                record = self.fl_engine.aggregate_models(
                    self.round_participants,
                    algorithm=self.fl_engine.algorithm,
                    round_number=self.current_round,
                )
                self.last_aggregation_record = record
                self.append_log(f"Aggregation sample counts: {record['client_sample_counts']}")
            self.current_step += 1
            QTimer.singleShot(1300, self.run_next_step)

        elif self.current_step == 6:
            if "Real" in self.mode_toggle.currentText():
                self.append_log("Evaluating new global model on distributed validation sets...")
                def _eval():
                    metrics = self.fl_engine.evaluate_global_model(self.trainer, round_number=self.current_round)
                    metrics["acc"] = metrics.get("accuracy", 0.0)
                    self.eval_done_signal.emit(metrics)
                threading.Thread(target=_eval, daemon=True).start()
            else:
                self.current_step += 1
                QTimer.singleShot(1300, self.run_next_step)

        elif self.current_step >= 7:
            for card in self.hospital_cards:
                if card.hospital_id in self.round_participants:
                    card.set_state("Global model received", "done")
            if self.current_step == len(self.step_sequence) - 1:
                self.step_widgets[self.current_step].set_status("done")
            self.current_step += 1
            QTimer.singleShot(1300, self.run_next_step)

        else:
            self.current_step += 1
            QTimer.singleShot(1300, self.run_next_step)
            
    def on_real_training_progress(self, r: dict):
        self.append_log(
            f"[{r['hospital_id']}] Epoch {r['epoch']} - loss={r['loss']:.4f}, "
            f"acc={r['acc']:.4f}, val_acc={r.get('val_acc', 0):.4f}, FN={r.get('false_negatives', 0)}"
        )

    def on_real_training_done(self, result: dict):
        if result.get("status") == "completed":
            self.append_log(
                f"[{result['hospital_id']}] Local update ready: samples={result.get('num_samples')}, "
                f"loss={result.get('local_loss'):.4f}, acc={result.get('local_accuracy'):.4f}."
            )
            self.db.save_client_update(
                project_id=self.current_project_id,
                round_number=self.current_round,
                hospital_id=result["hospital_id"],
                num_samples=int(result.get("num_samples", 0)),
                local_loss=result.get("local_loss"),
                local_accuracy=result.get("local_accuracy"),
                local_metrics=result.get("validation_metrics") or result.get("local_metrics"),
                status="completed",
            )
        else:
            self.append_log(f"[{result.get('hospital_id')}] Local training failed: {result.get('error')}")
            self.db.save_client_update(
                project_id=self.current_project_id,
                round_number=self.current_round,
                hospital_id=result.get("hospital_id"),
                num_samples=int(result.get("num_samples", 0)),
                local_loss=None,
                local_accuracy=None,
                local_metrics={"error": result.get("error")},
                status="failed",
            )
        self.active_train_threads -= 1
        if self.active_train_threads <= 0:
            self.current_step += 1
            self.run_next_step()
            
    def on_eval_done(self, r: dict):
        self.latest_global_metrics = dict(r)
        self.append_log(f"Global Model Evaluation - Loss: {r['loss']:.4f}, Acc: {r['acc']:.4f}")
        self.append_log(
            f"Medical metrics: F1={r.get('f1_score', 0):.4f}, "
            f"sensitivity={r.get('sensitivity', 0):.4f}, specificity={r.get('specificity', 0):.4f}, "
            f"FN={r.get('false_negatives', 0)}, FP={r.get('false_positives', 0)}."
        )
        self.db.log("training", f"Global Eval R{self.current_round}: Acc {r.get('accuracy', r.get('acc', 0)):.4f}", "success")
        self.db.save_evaluation_metrics(
            run_id=None,
            model_version_id=None,
            scope="project_runner_global",
            metrics=r,
            round_number=self.current_round,
            split="validation",
        )
        self.db.save_confusion_matrix(
            run_id=None,
            model_version_id=None,
            scope="project_runner_global",
            metrics=r,
            round_number=self.current_round,
            split="validation",
        )
        if self.last_aggregation_record:
            self.db.save_federated_round(
                project_id=self.current_project_id,
                round_number=self.current_round,
                aggregation_algorithm=self.last_aggregation_record.get("aggregation_algorithm", self.fl_engine.algorithm),
                participation_fraction=self.participation_fraction.value(),
                participating_clients=self.last_aggregation_record.get("participating_clients", []),
                client_sample_counts=self.last_aggregation_record.get("client_sample_counts", {}),
                global_metrics=r,
                status="completed",
            )
        self.current_step += 1
        self.run_next_step()

    def finalize_project_completion(self):
        if not self.current_project_id:
            return
        metrics = self.latest_global_metrics or {
            "status": "completed_visual_demo",
            "note": "Demo visual-only run; no real model evaluation was performed.",
        }
        final_model_path = None
        if self.fl_engine is not None and "Real" in self.mode_toggle.currentText():
            try:
                output_path = Path(self.config.get("models_dir", "models")) / "projects" / f"project_{self.current_project_id}_final.pt"
                final_model_path = self.fl_engine.save_global_checkpoint(output_path, metrics=metrics)
                self.append_log(f"Final global checkpoint saved: {final_model_path}")
            except Exception as exc:
                self.append_log(f"Final checkpoint could not be saved; Docker export will use a placeholder artifact. {exc}")

        self.db.complete_fl_project(
            self.current_project_id,
            final_model_path=final_model_path,
            final_metrics=metrics,
        )
        self.db.log("project", f"Project #{self.current_project_id} completed", "success")
        self.export_docker_btn.setEnabled(True)
        self.refresh_project_summary()

    def export_current_project_package(self):
        if not self.current_project_id:
            QMessageBox.warning(self, "Docker Export", "No FL project is selected.")
            return

        project = self.db.get_fl_project(self.current_project_id)
        if not project or str(project["status"]).lower() != "completed":
            QMessageBox.warning(self, "Docker Export", "Complete the FL project before exporting a Docker package.")
            return

        try:
            if self.user_role == "admin":
                members = self.db.list_joined_project_memberships(self.current_project_id)
                if not members:
                    QMessageBox.warning(self, "Docker Export", "No joined hospitals were found for this project.")
                    return
                exports = [
                    self.docker_exporter.export_for_hospital(
                        project_id=self.current_project_id,
                        hospital_id=str(member["hospital_id"]),
                        requester_role="admin",
                    )
                    for member in members
                ]
                paths = "\n".join(item["zip_path"] for item in exports)
                QMessageBox.information(self, "Docker Export Complete", f"Created Docker packages:\n{paths}")
            else:
                export = self.docker_exporter.export_for_hospital(
                    project_id=self.current_project_id,
                    hospital_id=self.logged_hospital_id,
                    requester_role="hospital",
                )
                QMessageBox.information(
                    self,
                    "Docker Export Complete",
                    f"Docker package created:\n{export['zip_path']}",
                )
        except DockerExportError as exc:
            QMessageBox.warning(self, "Docker Export", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Docker Export Error", f"Failed to export Docker package:\n{exc}")

    def reset_demo(self):
        self.current_project_id = None
        self.current_round = 1
        self.project_name.setText("Multi-center Pneumonia FL Study")
        self.summary_status.setText("Status: idle")
        self.reset_monitor_only()
        self.refresh_project_summary()

    def reset_monitor_only(self):
        self.demo_running = False
        self.current_step = 0
        self.progress.setValue(0)
        self.log_view.clear()
        self.explanation_text.setText("Create or request a project, then start the scenario.")

        for s in self.step_widgets:
            s.set_status("waiting")
        for c in self.hospital_cards:
            if self._is_hospital_active(c.hospital_id):
                c.set_state("Not selected", "idle")
            else:
                c.set_state("Inactive / unavailable", "unavailable")

    def selected_hospitals(self) -> List[str]:
        active_ids = self._active_hospital_ids()
        return [
            hospital_id
            for hospital_id, checkbox in self.hospital_checks.items()
            if checkbox.isChecked() and hospital_id in active_ids
        ]

    def _active_hospital_ids(self) -> set[str]:
        return {
            str(row["hospital_id"])
            for row in self.db.list_hospitals(active_only=True)
        }

    def _is_hospital_active(self, hospital_id: str) -> bool:
        hospital = self.db.get_hospital(hospital_id)
        return str(hospital["node_status"] if hospital else "inactive").lower() == "active"

    def _joined_memberships(self) -> list:
        if not self.current_project_id:
            return []
        return [
            row for row in self.db.list_project_memberships(self.current_project_id)
            if "joined" in str(row["status"]).lower()
        ]

    def append_log(self, text: str):
        self.log_view.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def refresh_project_summary(self):
        latest = self.db.get_latest_fl_project()
        if latest:
            self.current_project_id = int(latest["id"])
            self.summary_project.setText(latest["project_name"])
            
            import json
            details = {}
            if "details_json" in latest.keys() and latest["details_json"]:
                try:
                    details = json.loads(latest["details_json"])
                except:
                    pass
            
            creator = details.get('created_by_display_name')
            approver = details.get('approved_by')
            
            if creator:
                self.summary_owner.setText(f"Created by: {creator}")
            else:
                self.summary_owner.setText("Created by: Administration Department")
                
            if approver:
                self.summary_approved.setText(f"Approved by: {approver}")
            else:
                self.summary_approved.setText("Approved by: Administration Department")
                
            self.summary_goal.setText(f"Goal: {latest['disease_target']}")
            self.summary_backbone.setText(f"Backbone: {latest['model_backbone']}")
            self.summary_algo.setText(f"FL Algorithm: {latest['fl_algorithm']}")
            self.summary_round.setText(f"Current Round: {self.current_round}")
            self.summary_status.setText(f"Status: {latest['status'] or 'created'}")
            can_export = str(latest["status"] or "").lower() == "completed"
            if self.user_role == "hospital":
                membership = self.db.get_project_membership(int(latest["id"]), self.logged_hospital_id)
                can_export = can_export and bool(membership and "joined" in str(membership["status"]).lower())
            self.export_docker_btn.setEnabled(can_export)
            
            # Update hospital cards dynamically from project_memberships
            for i in reversed(range(self.hospital_cards_layout.count())): 
                w = self.hospital_cards_layout.itemAt(i).widget()
                if w: w.deleteLater()
            self.hospital_cards.clear()
            
            mems = self.db.list_project_memberships(self.current_project_id)
            for m in mems:
                card_h = HospitalStatusCard(m['hospital_id'], m['hospital_name'])
                hospital_active = self._is_hospital_active(m['hospital_id'])
                
                status_upper = m['status'].upper()
                if m['status'] == "requester_joined":
                    status_upper = "REQUESTER / JOINED"

                if not hospital_active:
                    card_h.set_state("INACTIVE / UNAVAILABLE", "unavailable")
                    self.hospital_cards.append(card_h)
                    self.hospital_cards_layout.addWidget(card_h)
                    continue
                    
                if "joined" in m['status']:
                    card_h.set_state(status_upper, "success")
                elif m['status'] == "invited":
                    card_h.set_state(status_upper, "waiting")
                elif m['status'] == "declined":
                    card_h.set_state(status_upper, "idle")
                else:
                    card_h.set_state(status_upper, "idle")
                    
                self.hospital_cards.append(card_h)
                self.hospital_cards_layout.addWidget(card_h)
                
        else:
            self.summary_project.setText("No active project")
            self.summary_owner.setText("Created by: -")
            self.summary_approved.setText("Approved by: -")
            self.summary_goal.setText("Goal: -")
            self.summary_backbone.setText("Backbone: -")
            self.summary_algo.setText("FL Algorithm: -")
            self.summary_round.setText("Current Round: -")
            self.summary_status.setText("Status: idle")
            self.export_docker_btn.setEnabled(False)

    def load_latest_project_to_ui(self):
        latest = self.db.get_latest_fl_project()
        if not latest: return
        
        # Populate basic fields
        self.project_name.setText(latest["project_name"])
        
        idx = self.goal_target.findText(latest["disease_target"])
        if idx >= 0: self.goal_target.setCurrentIndex(idx)
        else:
            self.goal_target.addItem(latest["disease_target"])
            self.goal_target.setCurrentText(latest["disease_target"])
            
        idx = self.model_backbone.findText(latest["model_backbone"])
        if idx >= 0: self.model_backbone.setCurrentIndex(idx)
        
        idx = self.fl_algorithm.findText(latest["fl_algorithm"])
        if idx >= 0: self.fl_algorithm.setCurrentIndex(idx)
        
        # Populate advanced fields
        self.total_rounds.setValue(latest["total_rounds"])
        self.local_epochs.setValue(latest["local_epochs"])
        self.batch_size.setValue(latest["batch_size"])
        self.learning_rate.setValue(latest["learning_rate"])
        self.participation_fraction.setValue(latest["participation_fraction"])
        self.stop_accuracy.setValue(latest["stop_accuracy"])
        
        # Populate hospital checkboxes
        if self.user_role == "admin":
            memberships = self.db.list_project_memberships(latest["id"])
            invited_ids = [m["hospital_id"] for m in memberships]
            for hid, cb in self.hospital_checks.items():
                if hid in invited_ids:
                    cb.setChecked(True)
                else:
                    cb.setChecked(False)
        
        self.refresh_project_summary()
