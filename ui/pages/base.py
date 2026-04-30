from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel, QFrame, QHBoxLayout, QVBoxLayout, QWidget, QScrollArea
)
from PySide6.QtCore import Qt


class BasePage(QWidget):
    def __init__(self, title: str, subtitle: str = "") -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar with title and subtitle
        top = QFrame()
        top.setObjectName("TopBar")
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(32, 20, 32, 20)
        top_layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("PageTitle")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("PageSubtitle")

        top_layout.addWidget(self.title_label)
        if subtitle:
            top_layout.addWidget(self.subtitle_label)

        root.addWidget(top)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setObjectName("ContentScroll")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #b9cbd8;
                border-radius: 5px;
                min-height: 40px;
            }
            QScrollBar::handle:vertical:hover {
                background: #8fa7b8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Main content widget
        content_widget = QWidget()
        content_widget.setObjectName("ContentWidget")
        self.body = QVBoxLayout(content_widget)
        self.body.setContentsMargins(32, 24, 32, 32)
        self.body.setSpacing(24)

        scroll.setWidget(content_widget)
        root.addWidget(scroll, 1)

    @property
    def layout(self):
        return self.body
