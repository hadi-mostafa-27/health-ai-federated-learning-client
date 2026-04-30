from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LoginWindow(QWidget):
    def __init__(self, on_login, db):
        super().__init__()
        self.on_login = on_login
        self.db = db
        self.setWindowTitle("Health AI Federated Learning")
        self.setObjectName("LoginWindow")

        root = QVBoxLayout(self)
        root.setContentsMargins(34, 32, 34, 32)
        root.setSpacing(18)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("PanelCard")
        card.setMinimumWidth(380)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 24, 26, 26)
        card_layout.setSpacing(16)

        title = QLabel("Health AI FL")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Sign in to manage federated learning workflows.")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        self.role = QComboBox()
        self.role.addItems(["Admin", "Hospital"])
        self.role.currentTextChanged.connect(self.on_role_changed)

        self.hospital = QComboBox()
        self.load_hospitals()

        self.username = QLineEdit()
        self.username.setPlaceholderText("Demo username")

        self.password = QLineEdit()
        self.password.setPlaceholderText("Demo password optional")
        self.password.setEchoMode(QLineEdit.Password)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setVerticalSpacing(12)
        form.addRow("Role", self.role)
        self.hosp_label = QLabel("Hospital")
        form.addRow(self.hosp_label, self.hospital)
        form.addRow("Username", self.username)
        form.addRow("Password", self.password)

        btn = QPushButton("Sign In")
        btn.setObjectName("PrimaryButton")
        btn.setMinimumHeight(42)
        btn.clicked.connect(self.login)

        footer = QLabel("Academic prototype. Not a clinical decision system.")
        footer.setObjectName("CardMeta")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addLayout(form)
        card_layout.addWidget(btn)
        card_layout.addWidget(footer)
        root.addWidget(card)

        self.on_role_changed(self.role.currentText())

    def load_hospitals(self):
        hospitals = self.db.list_hospitals(active_only=False)
        for hospital in hospitals:
            status = str(hospital["node_status"] or "active").lower()
            label = hospital["hospital_name"]
            if status != "active":
                label = f"{label} (inactive)"
            self.hospital.addItem(label, userData=hospital["hospital_id"])

    def on_role_changed(self, text):
        is_hospital = text.lower() == "hospital"
        self.hosp_label.setVisible(is_hospital)
        self.hospital.setVisible(is_hospital)

    def login(self):
        role = self.role.currentText().lower()
        if role == "admin":
            hospital_name = ""
            hospital_id = ""
            display_name = "Health Ministry / Administration Department"
        else:
            hospital_name = self.hospital.currentText()
            hospital_id = self.hospital.currentData()
            display_name = hospital_name

        self.on_login(role, hospital_name, hospital_id, display_name)
