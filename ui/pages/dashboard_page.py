"""
dashboard_page.py

Admin view:
  • Stat cards (existing)
  • Quick actions (existing)
  • ── NEW ── "FL Network Visualizer" panel:
      [Show Current Projects] → scrollable project list
      Click a project → animated node/particle diagram below

Hospital view:  identical to before (no network panel shown).
"""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage
from ui.widgets.fl_network_canvas import FLNetworkCanvas


# ─────────────────────── helper widgets ─────────────────────────────────────

class StatCard(QFrame):
    def __init__(self, title: str, value: str = "-", meta: str = "") -> None:
        super().__init__()
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("CardTitle")
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("CardValue")
        self.value_lbl.setWordWrap(True)
        self.meta_lbl = QLabel(meta)
        self.meta_lbl.setObjectName("CardMeta")
        self.meta_lbl.setWordWrap(True)

        layout.addWidget(title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.meta_lbl)
        layout.addStretch()

    def set_data(self, value: str, meta: str = "") -> None:
        self.value_lbl.setText(value)
        self.meta_lbl.setText(meta)


class ProjectCard(QFrame):
    """Clickable card representing one FL project in the project list."""

    STYLE_NORMAL = """
        QFrame {
            background: #ffffff;
            border: 1px solid #dbe7ef;
            border-radius: 8px;
        }
        QFrame:hover { border-color: #9bd8dc; background: #f6fbfc; }
    """
    STYLE_SELECTED = """
        QFrame {
            background: #eaf7f8;
            border: 2px solid #0f766e;
            border-radius: 8px;
        }
    """

    def __init__(self, project: dict, on_click) -> None:
        super().__init__()
        self._on_click = on_click
        self._project = project
        self.setStyleSheet(self.STYLE_NORMAL)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(138)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        name = project.get("project_name", "Unnamed")
        disease = project.get("disease_target", "")
        status = project.get("status", "created")
        backbone = project.get("model_backbone", "")
        created_by = project.get("created_by", "Administration Department")
        participants = project.get("participant_summary", "No joined hospitals")
        rounds = project.get("round_summary", "Rounds: -")
        final_metrics = project.get("final_metric_summary", "Final metrics: pending")
        docker = project.get("docker_summary", "Docker export: available after completion")

        name_lbl = QLabel(str(name))
        name_lbl.setStyleSheet("color:#102033; font-size:13px; font-weight:800;")

        meta_lbl = QLabel(f"{disease} | {backbone} | {rounds}")
        meta_lbl.setStyleSheet("color:#60758a; font-size:11px;")

        owner_lbl = QLabel(f"Created by: {created_by}")
        owner_lbl.setStyleSheet("color:#40566b; font-size:11px;")

        participant_lbl = QLabel(f"Hospitals: {participants}")
        participant_lbl.setStyleSheet("color:#40566b; font-size:11px;")

        status_colors = {
            "running": "#0f766e",
            "active":  "#0f766e",
            "created": "#b45309",
            "done":    "#60758a",
            "completed": "#0f766e",
            "stopped": "#b91c1c",
        }
        sc = status_colors.get(status.lower(), "#60758a")
        status_lbl = QLabel(f"Status: {status.upper()} | {final_metrics} | {docker}")
        status_lbl.setStyleSheet(f"color:{sc}; font-size:11px; font-weight:600;")

        layout.addWidget(name_lbl)
        layout.addWidget(meta_lbl)
        layout.addWidget(owner_lbl)
        layout.addWidget(participant_lbl)
        layout.addWidget(status_lbl)

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self._on_click(self._project)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self.STYLE_SELECTED if selected else self.STYLE_NORMAL)


# ─────────────────────────── main page ──────────────────────────────────────

class DashboardPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        user_role = config.get("user_role", "hospital")
        subtitle = (
            "Monitor hospital registry status, FL projects, and project requests."
            if user_role == "admin"
            else "Monitor this hospital node, local data, and recent activity."
        )
        super().__init__("Dashboard", subtitle)
        self.config = config
        self.db = db
        self._user_role = user_role
        self._project_cards: list[ProjectCard] = []
        self._selected_project_id: Optional[int] = None

        # ── stat cards ──────────────────────────────────────────────────────
        top_grid = QWidget()
        grid = QGridLayout(top_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(20)

        if self._user_role == "admin":
            self.cards = {
                "Administration": StatCard("Administration", "-", "System governance console"),
                "Hospitals": StatCard("Hospitals", "0/0", "Active hospitals of total registered"),
                "Latest FL Project": StatCard("Latest FL Project", "None", "No project created yet"),
                "Pending Requests": StatCard("Pending Requests", "0", "Project requests waiting for review"),
            }
        else:
            self.cards = {
                "Hospital": StatCard("Hospital", "-", "Node information"),
                "Local Samples": StatCard("Local Samples", "0", "Images available locally"),
                "Latest Model": StatCard("Latest Model", "None", "Active checkpoint"),
                "Last Prediction": StatCard("Last Prediction", "None", "Most recent inference"),
            }
        order = list(self.cards.keys())
        for idx, key in enumerate(order):
            grid.addWidget(self.cards[key], idx // 3, idx % 3)
        self.layout.addWidget(top_grid)

        # ── activity + node status ───────────────────────────────────────────
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(20)

        self.activity_card = QFrame()
        self.activity_card.setObjectName("PanelCard")
        activity_layout = QVBoxLayout(self.activity_card)
        activity_layout.setContentsMargins(24, 20, 24, 20)
        activity_layout.setSpacing(14)

        activity_title = QLabel("Recent Activity")
        activity_title.setObjectName("SectionTitle")
        activity_layout.addWidget(activity_title)

        self.activity_table = QTableWidget(0, 3)
        self.activity_table.setHorizontalHeaderLabels(["Time", "Type", "Description"])
        self.activity_table.verticalHeader().setVisible(False)
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.setAlternatingRowColors(True)
        self.activity_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.activity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.activity_table.setMaximumHeight(240)
        activity_layout.addWidget(self.activity_table)

        self.side_card = QFrame()
        self.side_card.setObjectName("PanelCard")
        side_layout = QVBoxLayout(self.side_card)
        side_layout.setContentsMargins(24, 20, 24, 20)
        side_layout.setSpacing(14)

        self.side_title = QLabel("Node Status")
        self.side_title.setObjectName("SectionTitle")
        side_layout.addWidget(self.side_title)

        self.status_badge = QLabel("ACTIVE")
        self.status_badge.setObjectName("BadgeSuccess")
        self.status_badge.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.status_badge)

        self.info_text = QLabel(
            "This hospital client can register datasets, run local prediction, "
            "generate Grad-CAM views, and participate in federated rounds."
        )
        self.info_text.setWordWrap(True)
        self.info_text.setObjectName("CardMeta")
        side_layout.addWidget(self.info_text)
        side_layout.addStretch(1)

        bottom_layout.addWidget(self.activity_card, 2)
        bottom_layout.addWidget(self.side_card, 1)
        self.layout.addWidget(bottom, 1)

        # ── FL Network Visualizer (admin only) ───────────────────────────────
        if self._user_role == "admin":
            self._build_network_panel()

        self.refresh()

    # ── FL network visualizer ────────────────────────────────────────────────

    def _build_network_panel(self) -> None:
        """Construct the collapsible FL network visualizer section."""

        # Outer panel card
        panel = QFrame()
        panel.setObjectName("PanelCard")
        panel.setStyleSheet("""
            #PanelCard {
                background: #ffffff;
                border: 1px solid #dbe7ef;
                border-radius: 8px;
            }
        """)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet("background: #f7fbfd; border-radius: 8px 8px 0 0; border-bottom: 1px solid #dbe7ef;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)

        title_lbl = QLabel("Federated Network Visualizer")
        title_lbl.setStyleSheet("color: #102033; font-size: 17px; font-weight: 800;")

        sub_lbl = QLabel("Shows active and inactive hospitals, selected project participants, and update flow.")
        sub_lbl.setStyleSheet("color: #60758a; font-size: 12px;")

        sub_w = QWidget()
        sub_col = QVBoxLayout(sub_w)
        sub_col.setContentsMargins(0, 0, 0, 0)
        sub_col.setSpacing(2)
        sub_col.addWidget(title_lbl)
        sub_col.addWidget(sub_lbl)

        self._show_projects_btn = QPushButton("Show Projects")
        self._show_projects_btn.setFixedHeight(40)
        self._show_projects_btn.setStyleSheet("""
            QPushButton {
                background: #0f766e;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                padding: 0 20px;
            }
            QPushButton:hover { background: #0b8a80; }
            QPushButton:pressed { background: #0b615b; }
        """)
        self._show_projects_btn.clicked.connect(self._toggle_project_list)

        self._refresh_viz_btn = QPushButton("Refresh")
        self._refresh_viz_btn.setFixedHeight(40)
        self._refresh_viz_btn.setToolTip("Reload hospital registry")
        self._refresh_viz_btn.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #0f766e;
                border: 1px solid #cfdde8;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover { background: #eaf7f8; border-color: #9bd8dc; }
        """)
        self._refresh_viz_btn.clicked.connect(self._reload_network)

        header_layout.addWidget(sub_w, 1)
        header_layout.addWidget(self._show_projects_btn)
        header_layout.addWidget(self._refresh_viz_btn)
        panel_layout.addWidget(header)

        # ── Body: project list (hidden) + canvas ─────────────────────────────
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Project list sidebar
        self._proj_sidebar = QFrame()
        self._proj_sidebar.setFixedWidth(300)
        self._proj_sidebar.setStyleSheet("background: #f7fbfd; border-right: 1px solid #dbe7ef;")
        self._proj_sidebar.setVisible(False)

        proj_layout = QVBoxLayout(self._proj_sidebar)
        proj_layout.setContentsMargins(12, 16, 12, 16)
        proj_layout.setSpacing(10)

        proj_title = QLabel("Available Projects")
        proj_title.setStyleSheet("color: #60758a; font-size: 11px; font-weight: 800;")
        proj_layout.addWidget(proj_title)

        # Scrollable project cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._proj_list_widget = QWidget()
        self._proj_list_widget.setStyleSheet("background: transparent;")
        self._proj_list_layout = QVBoxLayout(self._proj_list_widget)
        self._proj_list_layout.setContentsMargins(0, 0, 0, 0)
        self._proj_list_layout.setSpacing(8)
        self._proj_list_layout.addStretch()
        scroll.setWidget(self._proj_list_widget)
        proj_layout.addWidget(scroll)

        # Canvas area
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: transparent;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = FLNetworkCanvas()
        self._canvas.setMinimumHeight(520)
        canvas_layout.addWidget(self._canvas)

        body_layout.addWidget(self._proj_sidebar)
        body_layout.addWidget(canvas_container, 1)

        panel_layout.addWidget(body)
        self.layout.addWidget(panel)

        # Load hospitals into canvas
        self._reload_network()

    def _toggle_project_list(self) -> None:
        visible = self._proj_sidebar.isVisible()
        self._proj_sidebar.setVisible(not visible)
        if not visible:
            self._load_project_list()
            self._show_projects_btn.setText("Hide Projects")
        else:
            self._show_projects_btn.setText("Show Projects")

    def _load_project_list(self) -> None:
        """Fetch all FL projects and populate the sidebar cards."""
        # Clear old cards
        for card in self._project_cards:
            card.setParent(None)
        self._project_cards.clear()

        # Remove stretch, add cards, re-add stretch
        while self._proj_list_layout.count():
            item = self._proj_list_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        projects = self.db.fetchall("SELECT * FROM fl_projects ORDER BY id DESC")

        if not projects:
            empty = QLabel("No FL projects found.\nCreate one via Project Runner.")
            empty.setStyleSheet("color: #60758a; font-size: 12px;")
            empty.setAlignment(Qt.AlignCenter)
            self._proj_list_layout.addWidget(empty)
        else:
            for proj in projects:
                proj_dict = dict(proj)
                proj_dict.update(self._project_card_metadata(proj_dict))
                card = ProjectCard(proj_dict, self._on_project_selected)
                self._project_cards.append(card)
                self._proj_list_layout.addWidget(card)

        self._proj_list_layout.addStretch()

    def _project_card_metadata(self, project: dict) -> dict:
        details = {}
        if project.get("details_json"):
            try:
                details = json.loads(project.get("details_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                details = {}
        memberships = self.db.list_project_memberships(int(project["id"]))
        joined = [m for m in memberships if "joined" in str(m["status"]).lower()]
        inactive_joined = [
            m for m in joined
            if str((self.db.get_hospital(m["hospital_id"]) or {"node_status": "inactive"})["node_status"]).lower() != "active"
        ]
        participant_summary = f"{len(joined)} joined"
        if inactive_joined:
            participant_summary += f", {len(inactive_joined)} unavailable"

        metrics = self.db.get_latest_project_metrics(int(project["id"]))
        if str(project.get("status", "")).lower() == "completed":
            acc = metrics.get("accuracy", metrics.get("acc"))
            loss = metrics.get("loss")
            pieces = []
            if acc is not None:
                pieces.append(f"accuracy {float(acc):.3f}")
            if loss is not None:
                pieces.append(f"loss {float(loss):.3f}")
            final_metric_summary = "Final " + ", ".join(pieces) if pieces else "Final metrics saved"
            docker_summary = "Docker export: available"
        else:
            final_metric_summary = "Final metrics: pending"
            docker_summary = "Docker export: after completion"

        return {
            "created_by": details.get("created_by_display_name", "Administration Department"),
            "participant_summary": participant_summary,
            "round_summary": f"Rounds: {project.get('total_rounds', '-')}",
            "final_metric_summary": final_metric_summary,
            "docker_summary": docker_summary,
        }

    def _on_project_selected(self, project: dict) -> None:
        """User clicked a project card — visualise it."""
        pid = project.get("id")
        self._selected_project_id = pid

        # Highlight selected card
        for card in self._project_cards:
            card.set_selected(card._project.get("id") == pid)

        # Gather participant hospital IDs
        memberships = self.db.fetchall(
            "SELECT hospital_id, status FROM project_memberships WHERE project_id = ?", (pid,)
        )
        participant_ids = [
            dict(m)["hospital_id"]
            for m in memberships
            if "joined" in str(dict(m).get("status", "")).lower()
        ]

        # Fallback: parse from details_json if memberships empty
        if not participant_ids:
            details_json = project.get("details_json", "")
            if details_json:
                try:
                    details = json.loads(details_json)
                    requested = details.get("requested_hospitals", [])
                    for item in requested:
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            participant_ids.append(item[0])
                        elif isinstance(item, str):
                            participant_ids.append(item)
                except (json.JSONDecodeError, TypeError):
                    pass

        status = project.get("status", "created")
        self._canvas.set_project(
            project_name=project.get("project_name", "Unnamed"),
            project_status=status,
            participant_ids=participant_ids,
        )

    def _reload_network(self) -> None:
        """Load all registered hospitals into the canvas."""
        hospitals = self.db.list_hospitals(active_only=False)
        hospital_list = [dict(h) for h in hospitals]
        self._canvas.set_hospitals(hospital_list)
        if self._selected_project_id is not None:
            # Re-apply the selection
            proj = self.db.fetchone(
                "SELECT * FROM fl_projects WHERE id = ?", (self._selected_project_id,)
            )
            if proj:
                self._on_project_selected(dict(proj))

    # ── data refresh ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        if self._user_role == "admin":
            self._refresh_admin_cards()
            self.side_title.setText("Administration Status")
            self._set_badge("ADMIN", "BadgeSuccess")
            self.info_text.setText(
                "Administration monitors hospital availability, project requests, "
                "and FL project activity. Predictions and local training run from hospital workflows."
            )
        else:
            self._refresh_hospital_cards()
            self.side_title.setText("Node Status")
            h_id = self.config.get("hospital_id", "")
            h_profile = self.db.fetchone("SELECT * FROM hospital_profile WHERE hospital_id = ?", (h_id,))
            status = str(h_profile["node_status"] if h_profile else "active").upper()
            self._set_badge(status, "BadgeSuccess" if status == "ACTIVE" else "BadgeError")
            self.info_text.setText(
                "This hospital client can register datasets, run local prediction, "
                "generate Grad-CAM views, and participate in federated rounds."
            )

        rows = self.db.fetchall("SELECT * FROM activity_logs ORDER BY id DESC LIMIT 10")
        self.activity_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.activity_table.setItem(r, 0, QTableWidgetItem(str(row["created_at"])))
            self.activity_table.setItem(r, 1, QTableWidgetItem(str(row["event_type"])))
            self.activity_table.setItem(r, 2, QTableWidgetItem(str(row["description"])))
        self.activity_table.resizeColumnsToContents()

    def _refresh_admin_cards(self) -> None:
        total_hospitals = self._count("SELECT COUNT(*) AS count FROM hospital_profile")
        active_hospitals = self._count(
            "SELECT COUNT(*) AS count FROM hospital_profile WHERE node_status = 'active'"
        )
        pending_requests = self._count(
            "SELECT COUNT(*) AS count FROM project_requests WHERE status = 'pending'"
        )
        latest_project = self.db.get_latest_fl_project()

        self.cards["Administration"].set_data(
            self.config.get("display_name", "Administration Department"),
            "Project governance and hospital registry oversight",
        )
        self.cards["Hospitals"].set_data(
            f"{active_hospitals}/{total_hospitals}",
            "Active hospitals of total registered",
        )
        if latest_project:
            project_name = latest_project["project_name"] or "Unnamed project"
            details = [
                latest_project["disease_target"] or "No target",
                latest_project["fl_algorithm"] or latest_project["aggregation_algorithm"] or "FL algorithm not set",
                f"Status: {latest_project['status'] or 'created'}",
            ]
            self.cards["Latest FL Project"].set_data(project_name, " | ".join(details))
        else:
            self.cards["Latest FL Project"].set_data("None", "Create a project from FL Project Runner")
        self.cards["Pending Requests"].set_data(
            str(pending_requests),
            "Project requests waiting for admin review",
        )

    def _refresh_hospital_cards(self) -> None:
        latest_dataset = self.db.fetchone("SELECT * FROM datasets ORDER BY id DESC LIMIT 1")
        latest_model = self.db.fetchone("SELECT * FROM models ORDER BY id DESC LIMIT 1")
        latest_pred = self.db.fetchone("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")

        self.cards["Hospital"].set_data(
            f"{self.config.get('display_name', 'Health Node')}",
            f"{self.config.get('hospital_id', 'NODE')} | Network node",
        )
        self.cards["Local Samples"].set_data(
            str(latest_dataset["num_samples"] if latest_dataset else 0),
            "Registered local X-ray images",
        )
        self.cards["Latest Model"].set_data(
            latest_model["version"] if latest_model and latest_model["version"] else "None",
            latest_model["file_path"] if latest_model else "No active checkpoint recorded",
        )
        self.cards["Last Prediction"].set_data(
            latest_pred["predicted_label"] if latest_pred else "None",
            f"Confidence {latest_pred['confidence']:.2f}"
            if latest_pred and latest_pred["confidence"] is not None
            else "No inference yet",
        )

    def _count(self, query: str, params=()) -> int:
        row = self.db.fetchone(query, params)
        return int(row["count"] if row and row["count"] is not None else 0)

    def _set_badge(self, text: str, object_name: str) -> None:
        self.status_badge.setText(text)
        self.status_badge.setObjectName(object_name)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
