import json
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit,
    QSplitter, QWidget
)
from PySide6.QtCore import Qt
from core.config_manager import ConfigManager
from core.db import DatabaseManager
from core.docker_exporter import DockerExportError, DockerPackageExporter
from ui.pages.base import BasePage

class AvailableProjectsPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Project Invitations")
        self.config = config
        self.db = db
        self.exporter = DockerPackageExporter(db)
        self.hospital_id = self.config.get("hospital_id")
        
        self.layout.setSpacing(18)
        
        title = QLabel("Project Invitations")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")
        
        subtitle = QLabel("Review, accept, or decline federated learning project invitations.")
        subtitle.setStyleSheet("font-size:13px; color:#64748B;")
        
        self.layout.addWidget(title)
        self.layout.addWidget(subtitle)
        
        splitter = QSplitter(Qt.Vertical)
        
        # TOP: Table
        card_top = QFrame()
        card_top.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        top_layout = QVBoxLayout(card_top)
        top_layout.setContentsMargins(20, 20, 20, 20)
        
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Project ID", "Project Name", "Disease Target", "Algorithm", "Status", "My Response", "Docker Export"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.display_details)
        top_layout.addWidget(self.table)
        
        btns = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh List")
        self.accept_btn = QPushButton("Accept Invitation")
        self.decline_btn = QPushButton("Decline Invitation")
        self.export_btn = QPushButton("Export Docker Package")
        
        self.accept_btn.setStyleSheet("""
            QPushButton { background-color: #10B981; color: white; font-weight: bold; padding: 8px; border-radius: 6px; }
            QPushButton:hover { background-color: #059669; }
            QPushButton:disabled { background-color: #A7F3D0; }
        """)
        self.decline_btn.setStyleSheet("""
            QPushButton { background-color: #EF4444; color: white; font-weight: bold; padding: 8px; border-radius: 6px; }
            QPushButton:hover { background-color: #DC2626; }
            QPushButton:disabled { background-color: #FECACA; }
        """)
        
        self.refresh_btn.clicked.connect(self.load_invitations)
        self.accept_btn.clicked.connect(self.accept_invitation)
        self.decline_btn.clicked.connect(self.decline_invitation)
        self.export_btn.clicked.connect(self.export_package)
        
        btns.addWidget(self.refresh_btn)
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.decline_btn)
        btns.addWidget(self.export_btn)
        top_layout.addLayout(btns)
        
        splitter.addWidget(card_top)
        
        # BOTTOM: Details
        card_bottom = QFrame()
        card_bottom.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        bot_layout = QVBoxLayout(card_bottom)
        bot_layout.setContentsMargins(20, 20, 20, 20)
        
        bot_layout.addWidget(QLabel("<b>Project Details:</b>"))
        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        bot_layout.addWidget(self.details_view)
        
        splitter.addWidget(card_bottom)
        
        self.layout.addWidget(splitter)
        self.load_invitations()

    def load_invitations(self):
        self.table.setRowCount(0)
        self.projects_data = {}
        
        # Find memberships for this hospital
        memberships = self.db.fetchall("SELECT * FROM project_memberships WHERE hospital_id = ?", (self.hospital_id,))
        if not memberships: return
        
        for i, m in enumerate(memberships):
            pid = m['project_id']
            status = m['status']
            
            p = self.db.fetchone("SELECT * FROM fl_projects WHERE id = ?", (pid,))
            if not p: continue
            
            self.projects_data[pid] = {'project': p, 'membership': m}
            
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.table.setItem(i, 1, QTableWidgetItem(p['project_name']))
            self.table.setItem(i, 2, QTableWidgetItem(p['disease_target']))
            self.table.setItem(i, 3, QTableWidgetItem(p['fl_algorithm']))
            self.table.setItem(i, 4, QTableWidgetItem(p['status']))
            
            my_resp = QTableWidgetItem(status.upper())
            if status == "invited":
                my_resp.setForeground(Qt.blue)
            elif "joined" in status:
                my_resp.setForeground(Qt.darkGreen)
            elif status == "declined":
                my_resp.setForeground(Qt.red)
                
            self.table.setItem(i, 5, my_resp)
            can_export = p["status"] == "completed" and "joined" in status
            self.table.setItem(i, 6, QTableWidgetItem("Available" if can_export else "Not available"))
            
        self.display_details()

    def get_selected_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        return int(self.table.item(row, 0).text())

    def display_details(self):
        pid = self.get_selected_id()
        if not pid or pid not in self.projects_data:
            self.details_view.clear()
            self.accept_btn.setEnabled(False)
            self.decline_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            return
            
        data = self.projects_data[pid]
        p = data['project']
        m = data['membership']
        
        details = {}
        if p['details_json']:
            try:
                details = json.loads(p['details_json'])
            except:
                pass
                
        req_by = details.get('created_by_display_name', 'Admin')
        
        text = f"""
        <h3>{p['project_name']}</h3>
        <b>Requested By:</b> {req_by}<br>
        <b>Disease Target:</b> {p['disease_target']}<br>
        <b>Algorithm:</b> {p['fl_algorithm']}<br>
        <b>Backbone:</b> {p['model_backbone']}<br>
        <b>Rounds:</b> {p['total_rounds']}<br>
        <hr>
        <b>Project Status:</b> {p['status']}<br>
        <b>Your Membership:</b> {m['status'].upper()}<br>
        <b>Docker Export:</b> {'Available' if p['status'] == 'completed' and 'joined' in m['status'] else 'Available after project completion'}<br>
        """
        self.details_view.setHtml(text)
        
        # Only allow accept/decline if they are currently just "invited"
        if m['status'] == "invited":
            self.accept_btn.setEnabled(True)
            self.decline_btn.setEnabled(True)
        else:
            self.accept_btn.setEnabled(False)
            self.decline_btn.setEnabled(False)
        self.export_btn.setEnabled(p["status"] == "completed" and "joined" in m["status"])

    def accept_invitation(self):
        pid = self.get_selected_id()
        if not pid: return
        hospital = self.db.get_hospital(self.hospital_id)
        if str(hospital["node_status"] if hospital else "inactive").lower() != "active":
            QMessageBox.warning(self, "Inactive Hospital", "Inactive / unavailable hospitals cannot join new FL participation.")
            return
        self.db.update_membership_status(pid, self.hospital_id, "joined")
        self.db.log("project_invitation", f"{self.config.get('hospital_name')} accepted invite to Project #{pid}", "success")
        QMessageBox.information(self, "Accepted", "You have joined the project.")
        self.load_invitations()

    def decline_invitation(self):
        pid = self.get_selected_id()
        if not pid: return
        self.db.update_membership_status(pid, self.hospital_id, "declined")
        self.db.log("project_invitation", f"{self.config.get('hospital_name')} declined invite to Project #{pid}", "info")
        QMessageBox.information(self, "Declined", "You have declined the project invitation.")
        self.load_invitations()

    def export_package(self):
        pid = self.get_selected_id()
        if not pid:
            return
        try:
            export = self.exporter.export_for_hospital(
                project_id=pid,
                hospital_id=self.hospital_id,
                requester_role="hospital",
            )
            QMessageBox.information(
                self,
                "Docker Export Complete",
                f"Docker package created:\n{export['zip_path']}",
            )
            self.load_invitations()
        except DockerExportError as exc:
            QMessageBox.warning(self, "Docker Export", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Docker Export Error", f"Failed to export Docker package:\n{exc}")
