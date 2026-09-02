import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.styles import STYLE_SHEET

ICON_PATH = Path(__file__).resolve().parent / "app" / "assets" / "icon.ico"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MotionKey")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    app.setStyleSheet(STYLE_SHEET)

    window = MainWindow()
    window.resize(1240, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
