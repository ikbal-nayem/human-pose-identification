from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MotionKey")
        self.db = ConfigDatabase()

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.config_page = ConfigurationsPage(self.db)
        self.run_page = RunPage(self.db)
        self.stack.addWidget(self.run_page)
        self.stack.addWidget(self.config_page)
        layout.addWidget(self.stack, 1)

        self.nav_buttons[0].setChecked(True)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("Motion")
        title.setObjectName("appTitle")
        subtitle = QLabel("KEY")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

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

    def closeEvent(self, event):
        if self.run_page.worker is not None:
            self.run_page.worker.stop()
            self.run_page.worker.wait(2000)
        self.db.close()
        super().closeEvent(event)
