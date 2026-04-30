from PySide6.QtWidgets import (
    QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout
)
from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.base import BasePage
import sys

class ProfilePage(BasePage):
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        super().__init__("Profile & Session")
        self.config = config
        self.db = db
        
        self.layout.setSpacing(18)
        
        title = QLabel("My Profile")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F172A;")
        
        self.layout.addWidget(title)
        
        card = QFrame()
        card.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius:18px;")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        
        role = self.config.get("user_role", "hospital")
        
        if role == "admin":
            outer.addWidget(self._make_row("Role:", "System Administrator"))
            outer.addWidget(self._make_row("Display Name:", "Administration Department"))
            outer.addWidget(self._make_row("Permissions:", "Manage hospitals, approve requests, monitor FL"))
        else:
            h_id = self.config.get("hospital_id", "Unknown")
            h_name = self.config.get("hospital_name", "Unknown")
            
            # Fetch status
            h_profile = self.db.fetchone("SELECT * FROM hospital_profile WHERE hospital_id = ?", (h_id,))
            status = h_profile['node_status'] if h_profile else "active"
            self.current_status = status
            
            # Fetch stats
            joined = len(self.db.fetchall("SELECT * FROM project_memberships WHERE hospital_id = ? AND status = 'joined'", (h_id,)))
            pending = len(self.db.fetchall("SELECT * FROM project_requests WHERE hospital_name = ? AND status = 'pending'", (h_name,)))
            
            outer.addWidget(self._make_row("Role:", "Hospital Node"))
            outer.addWidget(self._make_row("Hospital Name:", h_name))
            outer.addWidget(self._make_row("Hospital ID:", h_id))
            
            self.status_label = QLabel(status.upper())
            self.status_label.setStyleSheet(f"color:{'#059669' if status == 'active' else '#DC2626'}; font-weight:bold;")
            status_row = QFrame()
            layout = QHBoxLayout(status_row)
            layout.setContentsMargins(0,0,0,0)
            lbl = QLabel("Status:")
            lbl.setStyleSheet("font-weight:bold; color:#64748B; min-width: 150px;")
            layout.addWidget(lbl)
            layout.addWidget(self.status_label)
            
            self.toggle_btn = QPushButton("Toggle Active/Inactive")
            self.toggle_btn.clicked.connect(self.toggle_status)
            layout.addWidget(self.toggle_btn)
            layout.addStretch()
            
            outer.addWidget(status_row)
            outer.addWidget(self._make_row("Joined Projects:", str(joined)))
            outer.addWidget(self._make_row("Pending Requests:", str(pending)))
            
        self.logout_btn = QPushButton("Logout / Switch User")
        self.logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #FEE2E2; 
                color: #991B1B; 
                border: 1px solid #FCA5A5;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FECACA;
            }
        """)
        self.logout_btn.clicked.connect(self.logout)
        
        outer.addStretch()
        outer.addWidget(self.logout_btn)
        
        self.layout.addWidget(card)
        self.layout.addStretch()

    def toggle_status(self):
        h_id = self.config.get("hospital_id")
        if not h_id: return
        
        new_status = "inactive" if self.current_status == "active" else "active"
        h_name = self.config.get("hospital_name", h_id)
        existing = self.db.fetchone("SELECT * FROM hospital_profile WHERE hospital_id = ?", (h_id,))
        if not existing:
            self.db.add_hospital(h_id, h_name, self.config.get("city", ""), new_status)
        else:
            self.db.update_hospital_status(h_id, new_status)
        self.db.log("hospital_profile", f"{h_name} changed node status to {new_status}", "warning" if new_status == "inactive" else "success")
        self.current_status = new_status
        self.status_label.setText(new_status.upper())
        self.status_label.setStyleSheet(f"color:{'#059669' if new_status == 'active' else '#DC2626'}; font-weight:bold;")

    def _make_row(self, label_text, value_text):
        row = QFrame()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0,0,0,0)
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-weight:bold; color:#64748B; min-width: 150px;")
        val = QLabel(value_text)
        val.setStyleSheet("color:#0F172A;")
        layout.addWidget(lbl)
        layout.addWidget(val, 1)
        return row

    def logout(self):
        # We clear session from config so it doesn't leak.
        self.config.set("user_role", "")
        self.config.set("hospital_name", "")
        self.config.set("hospital_id", "")
        self.config.set("display_name", "")
        
        # We need to restart the application to go back to the login screen.
        import os
        import subprocess
        # Restart the app. In PyInstaller, sys.executable is the packaged exe.
        if getattr(sys, "frozen", False):
            subprocess.Popen([sys.executable])
        else:
            subprocess.Popen([sys.executable, "app.py"])
        # Quit current instance
        sys.exit(0)
