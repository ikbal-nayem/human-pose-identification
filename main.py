import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.styles import STYLE_SHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Motion Control Studio")
    app.setStyleSheet(STYLE_SHEET)

    window = MainWindow()
    window.resize(1240, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
