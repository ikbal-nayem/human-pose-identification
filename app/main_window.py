import ctypes
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from db.database import ConfigDatabase
from app.config_page import ConfigurationsPage
from app.run_page import RunPage
from app.title_bar import TitleBar

BORDER = 6


class _ResizeGrip(QWidget):
    """Thin edge/corner strip that hands off to the OS system-resize."""

    def __init__(self, edges: Qt.Edge, cursor: Qt.CursorShape, kind: str, parent=None):
        super().__init__(parent)
        self._edges = edges
        self._kind = kind  # "h" fixed-height, "w" fixed-width, "corner" fixed both
        self.setObjectName("resizeGrip")
        self.setCursor(cursor)
        self.set_active(True)

    def set_active(self, active: bool):
        size = BORDER if active else 0
        if self._kind in ("h", "corner"):
            self.setFixedHeight(size)
        if self._kind in ("w", "corner"):
            self.setFixedWidth(size)
        self.setEnabled(active)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemResize(self._edges)
                event.accept()
                return
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MotionKey")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(900, 620)
        self.db = ConfigDatabase()

        self._build_frame()
        self.nav_buttons[0].setChecked(True)
        self._apply_rounded_corners()

    def _apply_rounded_corners(self):
        """Ask DWM (Windows 11) to round this frameless window's corners."""
        if sys.platform != "win32":
            return
        DWMWA_WINDOW_CORNER_PREFERENCE = 33
        DWMWCP_ROUND = 2
        try:
            preference = ctypes.c_int(DWMWCP_ROUND)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                int(self.winId()),
                DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(preference),
                ctypes.sizeof(preference),
            )
        except OSError:
            pass

    # ---- frameless window shell ------------------------------------------
    def _build_frame(self):
        outer = QWidget()
        outer.setObjectName("windowFrame")
        self.setCentralWidget(outer)
        grid = QGridLayout(outer)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        edge_specs = {
            "top": (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor, 0, 1, "h"),
            "bottom": (Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor, 2, 1, "h"),
            "left": (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor, 1, 0, "w"),
            "right": (Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor, 1, 2, "w"),
        }
        self._grips = []
        for edges, cursor, row, col, kind in edge_specs.values():
            grip = _ResizeGrip(edges, cursor, kind)
            grid.addWidget(grip, row, col)
            self._grips.append(grip)

        corner_specs = {
            (0, 0): (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeFDiagCursor),
            (0, 2): (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeBDiagCursor),
            (2, 0): (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeBDiagCursor),
            (2, 2): (Qt.Edge.BottomEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeFDiagCursor),
        }
        for (row, col), (edges, cursor) in corner_specs.items():
            grip = _ResizeGrip(edges, cursor, "corner")
            grid.addWidget(grip, row, col)
            self._grips.append(grip)

        content = QWidget()
        content.setObjectName("root")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.title_bar = TitleBar(self)
        content_layout.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.config_page = ConfigurationsPage(self.db)
        self.run_page = RunPage(self.db)
        self.stack.addWidget(self.run_page)
        self.stack.addWidget(self.config_page)
        body.addWidget(self.stack, 1)

        content_layout.addLayout(body, 1)
        grid.addWidget(content, 1, 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        section = QLabel("MENU")
        section.setObjectName("sidebarSection")
        layout.addWidget(section)

        self.nav_buttons: list[QPushButton] = []
        nav_items = [("▶  Run", 0), ("⚙  Configurations", 1)]
        for text, index in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked, i=index: self._navigate(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch(1)
        return sidebar

    def _navigate(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    # ---- keep the resize border usable across maximize/restore -----------
    def changeEvent(self, event):
        if event.type() == event.Type.WindowStateChange:
            maximized = self.isMaximized()
            for grip in self._grips:
                grip.set_active(not maximized)
            self.title_bar.refresh_maximize_glyph()
        super().changeEvent(event)

    def closeEvent(self, event):
        if self.run_page.worker is not None:
            self.run_page.worker.stop()
            self.run_page.worker.wait(2000)
        self.db.close()
        super().closeEvent(event)
