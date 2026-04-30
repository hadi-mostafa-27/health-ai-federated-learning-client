import json
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit,
    QSplitter
)
from PySide6.QtCore import Qt
from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage

class AdminRequestsPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Project Requests")
        self.config = config
        self.db = db
        
        self.layout.setSpacing(18)
        
        title = QLabel("Admin Approval: Project Requests")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")
        
        subtitle = QLabel("Review and approve full federated learning project setups proposed by hospitals.")
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
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Hospital", "Project Name", "Disease", 
            "Backbone", "Algo", "Status", "Date"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.display_details)
        top_layout.addWidget(self.table)
        
        btns = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.approve_btn = QPushButton("Approve Selected")
        self.reject_btn = QPushButton("Reject Selected")
        
        self.refresh_btn.clicked.connect(self.load_requests)
        self.approve_btn.clicked.connect(self.approve_request)
        self.reject_btn.clicked.connect(self.reject_request)
        
        btns.addWidget(self.refresh_btn)
        btns.addWidget(self.approve_btn)
        btns.addWidget(self.reject_btn)
        top_layout.addLayout(btns)
        
        splitter.addWidget(card_top)
        
        # BOTTOM: Details
        card_bottom = QFrame()
        card_bottom.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        bot_layout = QVBoxLayout(card_bottom)
        bot_layout.setContentsMargins(20, 20, 20, 20)
        
        bot_layout.addWidget(QLabel("<b>Selected Request Details:</b>"))
        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        bot_layout.addWidget(self.details_view)
        
        self.open_runner_btn = QPushButton("Open in FL Project Runner")
        self.open_runner_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.open_runner_btn.setVisible(False)
        self.open_runner_btn.clicked.connect(self.open_runner)
        bot_layout.addWidget(self.open_runner_btn)
        
        splitter.addWidget(card_bottom)
        
        self.layout.addWidget(splitter)
        self.load_requests()

    def load_requests(self):
        self.table.setRowCount(0)
        self.reqs_data = {}
        reqs = self.db.list_project_requests()
        for i, r in enumerate(reqs):
            self.reqs_data[r['id']] = r
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(r['id'])))
            self.table.setItem(i, 1, QTableWidgetItem(r['hospital_name']))
            self.table.setItem(i, 2, QTableWidgetItem(r['project_name']))
            self.table.setItem(i, 3, QTableWidgetItem(r['disease_target']))
            self.table.setItem(i, 4, QTableWidgetItem(r['suggested_backbone']))
            self.table.setItem(i, 5, QTableWidgetItem(r['fl_algorithm']))
            self.table.setItem(i, 6, QTableWidgetItem(r['status']))
            self.table.setItem(i, 7, QTableWidgetItem(str(r['created_at'])[:10]))
        self.details_view.clear()
        self.open_runner_btn.setVisible(False)

    def get_selected_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        return int(self.table.item(row, 0).text())

    def display_details(self):
        req_id = self.get_selected_id()
        if not req_id or req_id not in self.reqs_data:
            return
            
        r = self.reqs_data[req_id]
        
        details = {}
        if r['details_json']:
            try:
                details = json.loads(r['details_json'])
            except:
                pass
                
        hospitals_str = ", ".join([hname for hid, hname in details.get('requested_hospitals', [])])
        
        text = f"""
        <h3>{r['project_name']}</h3>
        <b>Requested By:</b> {r['hospital_name']}<br>
        <b>Disease Target:</b> {r['disease_target']}<br>
        <b>Task Type:</b> {details.get('task_type', 'N/A')}<br>
        <b>Dataset Schema:</b> {r['dataset_type']}<br>
        <b>Backbone:</b> {r['suggested_backbone']}<br>
        <b>Algorithm:</b> {r['fl_algorithm']}<br>
        <b>Requested Hospitals:</b> {hospitals_str}<br>
        """
        
        if r['status'] == 'approved':
            # Find the corresponding project
            proj = self.db.fetchone("SELECT id FROM fl_projects WHERE project_name = ? ORDER BY id DESC LIMIT 1", (r['project_name'],))
            if proj:
                pid = proj['id']
                mems = self.db.list_project_memberships(pid)
                joined = sum(1 for m in mems if 'joined' in m['status'])
                invited = sum(1 for m in mems if m['status'] == 'invited')
                declined = sum(1 for m in mems if m['status'] == 'declined')
                
                text += f"""
                <hr>
                <b>Active Project Setup:</b><br>
                - Project ID: #{pid}<br>
                - Joined/Requester: {joined}<br>
                - Pending Invitations: {invited}<br>
                - Declined: {declined}<br>
                """

        text += f"""
        <hr>
        <b>Advanced Settings:</b><br>
        - Rounds: {details.get('total_rounds', 'N/A')}<br>
        - Epochs: {details.get('local_epochs', 'N/A')}<br>
        - Batch Size: {details.get('batch_size', 'N/A')}<br>
        - Learning Rate: {details.get('learning_rate', 'N/A')}<br>
        - Participation Frac: {details.get('participation_fraction', 'N/A')}<br>
        - Target Accuracy: {details.get('stop_accuracy', 'N/A')}<br>
        <hr>
        <b>Clinical Motivation:</b><br>
        {r['reason']}
        """
        self.details_view.setHtml(text)
        
        if r['status'] == 'approved':
            self.open_runner_btn.setVisible(True)
        else:
            self.open_runner_btn.setVisible(False)

    def approve_request(self):
        req_id = self.get_selected_id()
        if not req_id:
            QMessageBox.warning(self, "Selection", "Please select a request first.")
            return
            
        project_id = self.db.approve_project_request(req_id)
        if project_id:
            self.db.log("project_approval", f"Admin approved request #{req_id} -> Project #{project_id}", "success")
            QMessageBox.information(self, "Approved", f"Request approved. New Project #{project_id} created.")
            self.load_requests()
            # Try to re-select the approved row
            for i in range(self.table.rowCount()):
                if int(self.table.item(i, 0).text()) == req_id:
                    self.table.selectRow(i)
                    break
        else:
            QMessageBox.warning(self, "Error", "Failed to create project.")

    def reject_request(self):
        req_id = self.get_selected_id()
        if not req_id:
            QMessageBox.warning(self, "Selection", "Please select a request first.")
            return
            
        self.db.reject_project_request(req_id)
        self.db.log("project_rejection", f"Admin rejected request #{req_id}", "warning")
        QMessageBox.information(self, "Rejected", f"Request #{req_id} rejected.")
        self.load_requests()

    def open_runner(self):
        # We need to switch the main window to the FL Project Runner page.
        # In main_window.py, the FL Project Runner is at a certain index.
        # We can find it by looking through the list widget items.
        main_win = self.window()
        if hasattr(main_win, 'nav'):
            for i in range(main_win.nav.count()):
                item = main_win.nav.item(i)
                if "FL Project Runner" in item.text():
                    main_win.nav.setCurrentRow(i)
                    
                    # Also force the runner to load the latest project settings
                    runner_page = main_win.stack.widget(i)
                    if hasattr(runner_page, 'refresh_project_summary'):
                        runner_page.load_latest_project_to_ui()
                    return
