from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QLineEdit, QComboBox, QFormLayout
)
from PySide6.QtGui import QBrush, QColor
from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage

class HospitalRegistryPage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Hospital Registry")
        self.config = config
        self.db = db
        
        self.layout.setSpacing(18)
        
        title = QLabel("Hospital Registry")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")
        
        subtitle = QLabel("Manage federated learning nodes.")
        subtitle.setStyleSheet("font-size:13px; color:#64748B;")
        
        self.layout.addWidget(title)
        self.layout.addWidget(subtitle)
        
        form_card = QFrame()
        form_card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        form_layout = QFormLayout(form_card)
        form_layout.setContentsMargins(20, 20, 20, 20)
        
        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("e.g. AUBMC")
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("e.g. American University of Beirut")
        self.input_city = QLineEdit()
        self.input_city.setPlaceholderText("e.g. Beirut")
        self.input_status = QComboBox()
        self.input_status.addItems(["active", "inactive"])
        
        form_layout.addRow("Hospital ID (Short):", self.input_id)
        form_layout.addRow("Hospital Name:", self.input_name)
        form_layout.addRow("City:", self.input_city)
        form_layout.addRow("Status:", self.input_status)
        
        self.add_btn = QPushButton("Add / Update Hospital")
        self.add_btn.clicked.connect(self.add_hospital)
        form_layout.addRow("", self.add_btn)
        
        self.layout.addWidget(form_card)
        
        table_card = QFrame()
        table_card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 20, 20, 20)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Hospital ID", "Hospital Name", "City", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        table_layout.addWidget(self.table)
        
        btns = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh List")
        self.remove_btn = QPushButton("Remove Selected")
        self.toggle_status_btn = QPushButton("Toggle Active/Inactive")
        
        self.refresh_btn.clicked.connect(self.load_hospitals)
        self.remove_btn.clicked.connect(self.remove_hospital)
        self.toggle_status_btn.clicked.connect(self.toggle_status)
        
        btns.addWidget(self.refresh_btn)
        btns.addWidget(self.remove_btn)
        btns.addWidget(self.toggle_status_btn)
        table_layout.addLayout(btns)
        
        self.layout.addWidget(table_card)
        
        self.seed_if_empty()
        self.load_hospitals()

    def seed_if_empty(self):
        hospitals = self.db.list_hospitals(active_only=False)
        if not hospitals:
            demo_hospitals = [
                ("AUBMC", "American University of Beirut Medical Center", "Beirut"),
                ("SGHUMC", "Saint George Hospital UMC", "Beirut"),
                ("HDF", "Hotel-Dieu de France", "Beirut"),
                ("LAUMC", "Lebanese American University Medical Center", "Beirut"),
                ("RHUH", "Rafik Hariri University Hospital", "Beirut"),
            ]
            for hid, name, city in demo_hospitals:
                self.db.add_hospital(hid, name, city, "active")

    def load_hospitals(self):
        self.table.setRowCount(0)
        hospitals = self.db.list_hospitals(active_only=False)
        for i, h in enumerate(hospitals):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(h['id'])))
            self.table.setItem(i, 1, QTableWidgetItem(h['hospital_id']))
            self.table.setItem(i, 2, QTableWidgetItem(h['hospital_name']))
            self.table.setItem(i, 3, QTableWidgetItem(h['city']))
            status_item = QTableWidgetItem("Active" if h['node_status'] == "active" else "Inactive / Unavailable")
            status_item.setForeground(QBrush(QColor("#047857" if h['node_status'] == "active" else "#B91C1C")))
            self.table.setItem(i, 4, status_item)

    def add_hospital(self):
        hid = self.input_id.text().strip().upper()
        name = self.input_name.text().strip()
        city = self.input_city.text().strip()
        status = self.input_status.currentText()
        
        if not hid or not name:
            QMessageBox.warning(self, "Error", "Hospital ID and Name are required.")
            return
            
        self.db.add_hospital(hid, name, city, status)
        self.db.log("hospital_registry", f"Added/Updated hospital: {name}", "success")
        self.input_id.clear()
        self.input_name.clear()
        self.input_city.clear()
        self.load_hospitals()

    def get_selected_hospital_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 1).text()

    def remove_hospital(self):
        hid = self.get_selected_hospital_id()
        if not hid:
            QMessageBox.warning(self, "Selection", "Select a hospital first.")
            return
            
        self.db.remove_hospital(hid)
        self.db.log("hospital_registry", f"Removed hospital: {hid}", "warning")
        self.load_hospitals()

    def toggle_status(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection", "Select a hospital first.")
            return
            
        hid = self.table.item(row, 1).text()
        current_status = self.table.item(row, 4).text().lower()
        new_status = "inactive" if "active" == current_status else "active"
        
        self.db.update_hospital_status(hid, new_status)
        self.load_hospitals()
