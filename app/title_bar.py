from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


class TitleBar(QWidget):
    """Custom dark title bar replacing the native OS chrome."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self._window = window
        self.setObjectName("titleBar")
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 0, 0)
        layout.setSpacing(8)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(str(ICON_PATH)).pixmap(18, 18))
        layout.addWidget(icon_label)

        text_label = QLabel("MotionKey")
        text_label.setObjectName("titleBarText")
        layout.addWidget(text_label)

        layout.addStretch(1)

        self.minimize_btn = self._make_button("─", "titleBarButton")
        self.minimize_btn.clicked.connect(self._window.showMinimized)

        self.maximize_btn = self._make_button("□", "titleBarButton")
        self.maximize_btn.clicked.connect(self._toggle_maximize)

        self.close_btn = self._make_button("✕", "titleBarCloseButton")
        self.close_btn.clicked.connect(self._window.close)

        for btn in (self.minimize_btn, self.maximize_btn, self.close_btn):
            layout.addWidget(btn)

    def _make_button(self, glyph: str, object_name: str) -> QPushButton:
        btn = QPushButton(glyph)
        btn.setObjectName(object_name)
        btn.setFixedSize(46, 40)
        btn.setCursor(Qt.CursorShape.ArrowCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return btn

    def _toggle_maximize(self):
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def refresh_maximize_glyph(self):
        self.maximize_btn.setText("▢" if self._window.isMaximized() else "□")

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._window.isMaximized():
            self._window.showNormal()
        handle = self._window.windowHandle()
        if handle is not None:
            handle.startSystemMove()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
