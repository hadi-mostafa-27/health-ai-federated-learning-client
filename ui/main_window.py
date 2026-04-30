from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from ui.pages.dashboard_page import DashboardPage
from ui.pages.dataset_page import DatasetPage
from ui.pages.federated_page import FederatedPage
from ui.pages.gradcam_page import GradCAMPage
from ui.pages.prediction_page import PredictionPage
from ui.pages.results_page import ResultsPage
from ui.pages.settings_page import SettingsPage
from ui.pages.training_page import TrainingPage
from ui.pages.project_runner_page import ProjectRunnerPage
from ui.pages.request_project_page import RequestProjectPage
from ui.pages.admin_requests_page import AdminRequestsPage
from ui.pages.available_projects_page import AvailableProjectsPage
from ui.pages.hospital_registry_page import HospitalRegistryPage
from ui.pages.profile_page import ProfilePage


class MainWindow(QMainWindow):
    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        super().__init__()
        self.config = config
        self.db = db
        self.setWindowTitle("Health AI Federated Learning Client")
        self.resize(1800, 1000)

        # Apply global stylesheet
        app = QApplication.instance()
        if app and not app.styleSheet():
            app.setStyleSheet(self._get_stylesheet())

        central = QWidget()
        central.setObjectName("central")
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = self._build_sidebar()
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        user_role = self.config.get("user_role", "admin")
        
        if user_role == "admin":
            pages_data = [
                ("Dashboard", DashboardPage(config, db)),
                ("Hospital Registry", HospitalRegistryPage(config, db)),
                ("Project Requests", AdminRequestsPage(config, db)),
                ("FL Project Runner", ProjectRunnerPage(config, db)),
                ("FL Sync", FederatedPage(config, db)),
                ("Results", ResultsPage(config, db)),
                ("Profile", ProfilePage(config, db)),
            ]
        else:
            pages_data = [
                ("Dashboard", DashboardPage(config, db)),
                ("Request Project", RequestProjectPage(config, db)),
                ("Project Invitations", AvailableProjectsPage(config, db)),
                ("Dataset", DatasetPage(config, db)),
                ("Prediction", PredictionPage(config, db)),
                ("Grad-CAM", GradCAMPage(config, db)),
                ("Local Training", TrainingPage(config, db)),
                ("FL Sync", FederatedPage(config, db)),
                ("Results", ResultsPage(config, db)),
                ("Profile", ProfilePage(config, db)),
            ]

        for name, page in pages_data:
            item = QListWidgetItem(name)
            item.setSizeHint(item.sizeHint() + QSize(0, 8))
            self.nav.addItem(item)
            self.stack.addWidget(page)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        outer.addWidget(self.sidebar)
        outer.addWidget(self.stack, 1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(20)

        # Brand area
        brand_title = QLabel("Health AI FL")
        brand_title.setObjectName("BrandTitle")
        brand_subtitle = QLabel("Federated Learning Client")
        brand_subtitle.setObjectName("BrandSubtitle")

        layout.addWidget(brand_title)
        layout.addWidget(brand_subtitle)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: rgba(255,255,255,0.15); max-height: 1px;")
        layout.addWidget(sep)

        # Navigation
        nav_label = QLabel("NAVIGATION")
        nav_label.setObjectName("NavLabel")
        layout.addWidget(nav_label)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setSpacing(6)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        layout.addWidget(self.nav, 1)

        # Footer separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: rgba(255,255,255,0.15); max-height: 1px;")
        layout.addWidget(sep2)

        # Hospital info footer
        hospital_info = QLabel(
            f"{self.config.get('hospital_name', 'Hospital')}\n{self.config.get('hospital_id', 'NODE')}"
        )
        hospital_info.setObjectName("HospitalFooter")
        hospital_info.setWordWrap(True)
        hospital_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(hospital_info)

        return sidebar

    def _get_stylesheet(self) -> str:
        return """
        /* GLOBAL STYLES */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        }
        QMainWindow {
            background-color: #f8fafc;
        }
        QWidget {
            background-color: transparent;
            color: #0f172a;
        }

        /* SIDEBAR */
        #Sidebar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0d2d4f, stop:1 #1a3f5f);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        #BrandTitle {
            color: white;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        #BrandSubtitle {
            color: rgba(255,255,255,0.7);
            font-size: 12px;
            font-weight: 500;
            margin-top: -4px;
        }
        #NavLabel {
            color: rgba(255,255,255,0.5);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        #HospitalFooter {
            background: rgba(0,0,0,0.25);
            border-radius: 14px;
            padding: 14px 16px;
            color: rgba(255,255,255,0.85);
            font-size: 12px;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.08);
        }

        /* NAVIGATION LIST */
        #NavList {
            background: transparent;
            border: none;
            outline: none;
        }
        #NavList::item {
            background: rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 2px 0;
            color: rgba(255,255,255,0.75);
            font-weight: 500;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.03);
        }
        #NavList::item:selected {
            background: rgba(255,255,255,0.18);
            color: white;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
        }
        #NavList::item:hover:!selected {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.08);
        }

        /* TOP BAR */
        #TopBar {
            background: white;
            border-bottom: 1px solid #e2e8f0;
        }
        #PageTitle {
            font-size: 32px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -1px;
        }
        #PageSubtitle {
            font-size: 14px;
            color: #64748b;
            font-weight: 400;
            margin-top: 2px;
        }

        /* SCROLL AREA */
        #ContentScroll {
            background: #f8fafc;
        }

        /* CARDS */
        #StatCard {
            background: white;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        #StatCard:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        #PanelCard {
            background: white;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        #CardTitle {
            color: #94a3b8;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        #CardValue {
            font-size: 36px;
            font-weight: 800;
            color: #0d2d4f;
            letter-spacing: -1px;
            margin: 8px 0;
        }
        #CardMeta {
            color: #64748b;
            font-size: 13px;
            font-weight: 400;
            line-height: 1.5;
        }
        #SectionTitle {
            font-size: 20px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }

        /* BUTTONS */
        QPushButton {
            background-color: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 11px 20px;
            font-weight: 600;
            color: #334155;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #e2e8f0;
            border-color: #cbd5e1;
        }
        QPushButton:pressed {
            background-color: #cbd5e1;
        }
        #PrimaryButton {
            background-color: #0d2d4f;
            color: white;
            border: none;
            font-weight: 600;
        }
        #PrimaryButton:hover {
            background-color: #1a3f5f;
        }
        #PrimaryButton:pressed {
            background-color: #0a1e35;
        }

        /* TABLES */
        QTableWidget {
            background: white;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            gridline-color: #f1f5f9;
        }
        QTableWidget::item {
            padding: 12px;
            color: #334155;
        }
        QHeaderView::section {
            background-color: #f8fafc;
            padding: 12px;
            font-weight: 700;
            color: #64748b;
            border: none;
            border-bottom: 1px solid #e2e8f0;
            font-size: 12px;
        }

        /* INPUTS */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px 14px;
            color: #0f172a;
            font-size: 13px;
            selection-background-color: #0d2d4f;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
            border: 1px solid #0d2d4f;
            outline: none;
        }
        QPlainTextEdit {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px;
            color: #0f172a;
            font-size: 12px;
        }

        /* PROGRESS BAR */
        QProgressBar {
            background: #e2e8f0;
            border-radius: 8px;
            height: 8px;
            border: none;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0d2d4f, stop:1 #1a3f5f);
            border-radius: 8px;
        }

        /* BADGES */
        #BadgeSuccess {
            background: #dcfce7;
            color: #166534;
            font-weight: 700;
            border-radius: 20px;
            padding: 8px 14px;
            font-size: 12px;
        }
        #BadgeWarning {
            background: #fef3c7;
            color: #92400e;
            font-weight: 700;
            border-radius: 20px;
            padding: 8px 14px;
            font-size: 12px;
        }
        #BadgeError {
            background: #fee2e2;
            color: #991b1b;
            font-weight: 700;
            border-radius: 20px;
            padding: 8px 14px;
            font-size: 12px;
        }

        /* GROUP BOX */
        QGroupBox {
            background: white;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            margin-top: 16px;
            font-weight: 700;
            color: #0f172a;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 10px 0 10px;
        }

        /* CHECKBOX */
        QCheckBox {
            spacing: 10px;
            color: #334155;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 5px;
            border: 2px solid #cbd5e1;
            background: white;
        }
        QCheckBox::indicator:checked {
            background: #0d2d4f;
            border: 2px solid #0d2d4f;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #94a3b8;
        }

        /* RADIO BUTTON */
        QRadioButton {
            spacing: 10px;
            color: #334155;
        }
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 2px solid #cbd5e1;
        }
        QRadioButton::indicator:checked {
            background: #0d2d4f;
            border: 2px solid #0d2d4f;
        }

        QMessageBox, QDialog {
            background: #ffffff;
            color: #0f172a;
        }
        QMessageBox QLabel, QDialog QLabel {
            color: #0f172a;
            background: transparent;
        }
        QMessageBox QPushButton, QDialog QPushButton {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 8px 18px;
            color: #0f172a;
            font-weight: 700;
            min-width: 90px;
        }
        QMessageBox QPushButton:hover, QDialog QPushButton:hover {
            background: #e2e8f0;
        }
        """
