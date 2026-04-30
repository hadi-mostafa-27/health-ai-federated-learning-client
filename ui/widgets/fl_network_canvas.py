from __future__ import annotations

import math
from typing import Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class FLNetworkCanvas(QWidget):
    """Visual overview of hospitals, project participation, and FL update flow."""

    MAX_VISIBLE_NODES = 24

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hospitals: list[dict] = []
        self._project_name = "No project selected"
        self._project_status = "idle"
        self._participant_ids: set[str] = set()
        self._phase = 0.0

        self.setMinimumSize(680, 440)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._timer = QTimer(self)
        self._timer.setInterval(42)
        self._timer.timeout.connect(self._advance_animation)
        self._timer.start()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(980, 540)

    def set_hospitals(self, hospitals: Sequence[Mapping[str, object]] | None) -> None:
        self._hospitals = [dict(hospital) for hospital in (hospitals or [])]
        self.update()

    def set_project(
        self,
        project_name: str,
        project_status: str,
        participant_ids: Sequence[str] | None = None,
    ) -> None:
        self._project_name = project_name or "Unnamed project"
        self._project_status = project_status or "unknown"
        self._participant_ids = {str(pid) for pid in (participant_ids or []) if pid}
        self.update()

    def _advance_animation(self) -> None:
        self._phase = (self._phase + 0.012) % 1.0
        if self.isVisible():
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            self._draw(painter)
        finally:
            painter.end()

    def _draw(self, painter: QPainter) -> None:
        rect = self.rect()
        painter.fillRect(rect, QColor("#f7fbfd"))
        self._draw_background_grid(painter)
        self._draw_header(painter)

        hospitals = self._visible_hospitals()
        if not hospitals:
            self._draw_empty_state(painter)
            return

        center = QPointF(rect.width() * 0.54, rect.height() * 0.53)
        radius = max(120.0, min(rect.width(), rect.height()) * 0.34)
        positions = self._node_positions(center, radius, len(hospitals))

        for index, hospital in enumerate(hospitals):
            self._draw_link(painter, center, positions[index], hospital)

        self._draw_server(painter, center)

        for index, hospital in enumerate(hospitals):
            self._draw_hospital_node(painter, positions[index], hospital)

        self._draw_legend(painter, rect)
        self._draw_footer(painter, rect, hospitals)

    def _draw_background_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#e4eef4"), 1))
        step = 40
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)

    def _draw_header(self, painter: QPainter) -> None:
        margin = 24
        painter.setPen(QColor("#60758a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(margin, 30, "Federated Aggregation Prototype")

        painter.setPen(QColor("#102033"))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title = self._elide(painter, self._project_name, max(180, self.width() - 360))
        painter.drawText(margin, 60, title)

        status_text = f"Status: {self._project_status.upper()}"
        badge_rect = QRectF(self.width() - 196, 22, 158, 34)
        painter.setPen(QPen(QColor("#cfdde8"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(badge_rect, 8, 8)
        painter.setPen(self._status_color(self._project_status))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, status_text)

    def _draw_empty_state(self, painter: QPainter) -> None:
        center = self.rect().center()
        painter.setPen(QColor("#60758a"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(center.x() - 240, center.y() - 28, 480, 56),
            Qt.AlignmentFlag.AlignCenter,
            "No hospitals are registered yet",
        )

    def _draw_link(self, painter: QPainter, center: QPointF, node: QPointF, hospital: Mapping[str, object]) -> None:
        participant = self._is_participant(hospital)
        color = QColor("#0f766e") if participant else QColor("#c8d7e1")
        painter.setPen(QPen(color, 2 if participant else 1))
        painter.drawLine(center, node)

        if participant:
            outgoing = self._point_between(center, node, self._phase)
            incoming = self._point_between(node, center, (self._phase + 0.48) % 1.0)
            self._draw_particle(painter, outgoing, QColor("#0ea5e9"))
            self._draw_particle(painter, incoming, QColor("#14b8a6"))

    def _draw_server(self, painter: QPainter, center: QPointF) -> None:
        painter.setPen(QPen(QColor("#075985"), 2))
        painter.setBrush(QColor("#e0f2fe"))
        painter.drawEllipse(center, 50, 50)

        painter.setPen(QColor("#0c4a6e"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(center.x() - 46, center.y() - 19, 92, 22),
            Qt.AlignmentFlag.AlignCenter,
            "Global",
        )
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(center.x() - 54, center.y() + 4, 108, 24),
            Qt.AlignmentFlag.AlignCenter,
            "Server",
        )

    def _draw_hospital_node(self, painter: QPainter, pos: QPointF, hospital: Mapping[str, object]) -> None:
        hospital_id = str(hospital.get("hospital_id", "NODE"))
        hospital_name = str(hospital.get("hospital_name", hospital_id))
        status = self._node_status(hospital)
        participant = self._is_participant(hospital)

        fill, border, text_color = self._node_colors(status)
        radius = 27 if participant else 23

        if participant:
            painter.setPen(QPen(QColor("#0ea5e9"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(pos, radius + 5, radius + 5)

        painter.setPen(QPen(border, 2))
        painter.setBrush(fill)
        painter.drawEllipse(pos, radius, radius)

        dot_center = QPointF(pos.x() + radius * 0.72, pos.y() - radius * 0.72)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(border)
        painter.drawEllipse(dot_center, 6, 6)

        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(
            QRectF(pos.x() - radius, pos.y() - 9, radius * 2, 18),
            Qt.AlignmentFlag.AlignCenter,
            self._initials(hospital_id),
        )

        label_width = 138
        label = self._elide(painter, hospital_name, label_width)
        painter.setPen(QColor("#334155") if participant else QColor("#60758a"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(pos.x() - label_width / 2, pos.y() + radius + 7, label_width, 30),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            label,
        )

    def _draw_particle(self, painter: QPainter, point: QPointF, color: QColor) -> None:
        glow = QColor(color)
        glow.setAlpha(70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(point, 7.0, 7.0)
        painter.setBrush(color)
        painter.drawEllipse(point, 3.5, 3.5)

    def _draw_legend(self, painter: QPainter, rect) -> None:
        x = 24
        y = rect.height() - 76
        self._draw_legend_item(painter, x, y, QColor("#16a34a"), "Active hospital")
        self._draw_legend_item(painter, x + 148, y, QColor("#dc2626"), "Inactive hospital")
        self._draw_legend_item(painter, x + 316, y, QColor("#0ea5e9"), "Project participant")

    def _draw_legend_item(self, painter: QPainter, x: int, y: int, color: QColor, text: str) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(x + 7, y + 7), 6, 6)
        painter.setPen(QColor("#40566b"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(x + 20, y + 12, text)

    def _draw_footer(self, painter: QPainter, rect, hospitals: Sequence[Mapping[str, object]]) -> None:
        total = len(self._hospitals)
        shown = len(hospitals)
        active = sum(1 for hospital in self._hospitals if self._node_status(hospital) == "active")
        inactive = total - active
        participating = sum(1 for hospital in hospitals if self._is_participant(hospital))
        unavailable_selected = sum(
            1
            for hospital in hospitals
            if str(hospital.get("hospital_id", "")) in self._participant_ids
            and self._node_status(hospital) != "active"
        )
        text = (
            f"Showing {shown} of {total} hospitals | Active: {active} | "
            f"Inactive: {inactive} | Active participants: {participating} | "
            f"Unavailable selected: {unavailable_selected}"
        )
        painter.setPen(QColor("#60758a"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(24, rect.height() - 36, rect.width() - 48, 22), text)

    def _visible_hospitals(self) -> list[dict]:
        if not self._hospitals:
            return []
        if self._participant_ids:
            participants = [
                hospital for hospital in self._hospitals
                if str(hospital.get("hospital_id", "")) in self._participant_ids
            ]
            others = [
                hospital for hospital in self._hospitals
                if str(hospital.get("hospital_id", "")) not in self._participant_ids
            ]
            return (participants + others)[: self.MAX_VISIBLE_NODES]
        return self._hospitals[: self.MAX_VISIBLE_NODES]

    def _node_positions(self, center: QPointF, radius: float, count: int) -> list[QPointF]:
        if count == 1:
            return [QPointF(center.x(), center.y() - radius)]
        start_angle = -math.pi / 2
        return [
            QPointF(
                center.x() + math.cos(start_angle + (2 * math.pi * idx / count)) * radius,
                center.y() + math.sin(start_angle + (2 * math.pi * idx / count)) * radius,
            )
            for idx in range(count)
        ]

    def _is_participant(self, hospital: Mapping[str, object]) -> bool:
        return (
            bool(self._participant_ids)
            and str(hospital.get("hospital_id", "")) in self._participant_ids
            and self._node_status(hospital) == "active"
        )

    @staticmethod
    def _node_status(hospital: Mapping[str, object]) -> str:
        status = str(hospital.get("node_status", hospital.get("status", "active"))).lower()
        return "active" if status == "active" else "inactive"

    @staticmethod
    def _node_colors(status: str) -> tuple[QColor, QColor, QColor]:
        if status == "active":
            return QColor("#16a34a"), QColor("#15803d"), QColor("#ffffff")
        return QColor("#dc2626"), QColor("#991b1b"), QColor("#ffffff")

    @staticmethod
    def _point_between(start: QPointF, end: QPointF, fraction: float) -> QPointF:
        return QPointF(
            start.x() + (end.x() - start.x()) * fraction,
            start.y() + (end.y() - start.y()) * fraction,
        )

    @staticmethod
    def _initials(value: str) -> str:
        clean = "".join(ch for ch in value if ch.isalnum())
        return (clean[:3] or "H").upper()

    @staticmethod
    def _status_color(status: str) -> QColor:
        colors = {
            "running": QColor("#0f766e"),
            "active": QColor("#0f766e"),
            "created": QColor("#b45309"),
            "done": QColor("#60758a"),
            "completed": QColor("#60758a"),
            "stopped": QColor("#b91c1c"),
            "failed": QColor("#b91c1c"),
        }
        return colors.get((status or "").lower(), QColor("#60758a"))

    @staticmethod
    def _elide(painter: QPainter, text: str, width: int) -> str:
        return painter.fontMetrics().elidedText(str(text), Qt.TextElideMode.ElideRight, width)
