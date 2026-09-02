STYLE_SHEET = """
* {
    font-family: 'Segoe UI', sans-serif;
    outline: none;
}

QMainWindow, QWidget#root {
    background-color: #191b22;
}

QWidget {
    color: #e9ebf3;
}

/* ---- sidebar --------------------------------------------------------- */
QWidget#sidebar {
    background-color: #14151b;
    border-right: 1px solid #262835;
}

QLabel#appTitle {
    color: #ffffff;
    font-size: 17px;
    font-weight: 600;
    padding: 22px 18px 4px 18px;
}

QLabel#appSubtitle {
    color: #767c92;
    font-size: 11px;
    padding: 0px 18px 18px 18px;
}

QPushButton#navButton {
    text-align: left;
    padding: 12px 18px;
    border: none;
    border-radius: 0px;
    background: transparent;
    color: #b7bbcc;
    font-size: 13px;
    font-weight: 500;
}

QPushButton#navButton:hover {
    background-color: #1f212b;
    color: #ffffff;
}

QPushButton#navButton:checked {
    background-color: #232538;
    color: #8f7cff;
    border-left: 3px solid #8f7cff;
}

/* ---- generic surfaces ------------------------------------------------- */
QWidget#page {
    background-color: #191b22;
}

QFrame#card {
    background-color: #1e2029;
    border: 1px solid #2a2c3a;
    border-radius: 12px;
}

QLabel#pageTitle {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
}

QLabel#pageHint {
    color: #868ca3;
    font-size: 12px;
}

QLabel#sectionLabel {
    color: #9aa0b8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}

/* ---- buttons ------------------------------------------------------------ */
QPushButton {
    background-color: #262838;
    color: #e9ebf3;
    border: 1px solid #34364a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #2f3245;
    border-color: #454868;
}

QPushButton:pressed {
    background-color: #21232f;
}

QPushButton:disabled {
    color: #565a6c;
    background-color: #1e2029;
    border-color: #262835;
}

QPushButton#primaryButton {
    background-color: #7c5cff;
    border: 1px solid #7c5cff;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #8f72ff;
}

QPushButton#dangerButton {
    background-color: transparent;
    border: 1px solid #4a2b34;
    color: #ff7c92;
}

QPushButton#dangerButton:hover {
    background-color: #2c1820;
}

QPushButton#startButton {
    background-color: #22c98f;
    border: 1px solid #22c98f;
    color: #0d1f18;
    font-weight: 700;
    font-size: 14px;
    padding: 12px 22px;
    border-radius: 10px;
}

QPushButton#startButton:hover {
    background-color: #34e0a3;
}

QPushButton#stopButton {
    background-color: #ff5470;
    border: 1px solid #ff5470;
    color: #2a0810;
    font-weight: 700;
    font-size: 14px;
    padding: 12px 22px;
    border-radius: 10px;
}

QPushButton#stopButton:hover {
    background-color: #ff6c85;
}

QPushButton#keyCapture {
    background-color: #232538;
    border: 1px solid #3b3e58;
    border-radius: 6px;
    color: #cfd2e6;
    padding: 5px 10px;
    min-width: 90px;
}

QPushButton#keyCapture:hover {
    border-color: #7c5cff;
    color: #ffffff;
}

QPushButton#clearKey {
    background: transparent;
    border: none;
    color: #6b7086;
    padding: 4px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton#clearKey:hover {
    color: #ff7c92;
}

/* ---- inputs -------------------------------------------------------------- */
QLineEdit, QComboBox {
    background-color: #1e2029;
    border: 1px solid #34364a;
    border-radius: 8px;
    padding: 7px 10px;
    color: #e9ebf3;
    font-size: 12px;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #7c5cff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #20222d;
    border: 1px solid #34364a;
    selection-background-color: #322f57;
    color: #e9ebf3;
    outline: none;
}

/* ---- lists / tables -------------------------------------------------------- */
QListWidget {
    background-color: #1e2029;
    border: 1px solid #2a2c3a;
    border-radius: 10px;
    padding: 4px;
    font-size: 13px;
}

QListWidget::item {
    padding: 10px 10px;
    border-radius: 6px;
    margin: 2px;
}

QListWidget::item:selected {
    background-color: #322f57;
    color: #ffffff;
}

QListWidget::item:hover:!selected {
    background-color: #262838;
}

QTableWidget {
    background-color: #1e2029;
    border: 1px solid #2a2c3a;
    border-radius: 10px;
    gridline-color: #262835;
    font-size: 12px;
    selection-background-color: #262838;
}

QHeaderView::section {
    background-color: #1a1c25;
    color: #9aa0b8;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #2a2c3a;
    font-size: 11px;
    font-weight: 600;
}

QTableWidget::item {
    padding: 4px;
}


QLabel#videoLabel {
    background-color: #000000;
    border: 1px solid #2a2c3a;
    border-radius: 12px;
}

QLabel#statusBadge {
    background-color: #232538;
    border: 1px solid #3b3e58;
    border-radius: 12px;
    padding: 4px 10px;
    color: #8f7cff;
    font-size: 11px;
    font-weight: 600;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
}

QScrollBar::handle:vertical {
    background: #34364a;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #454868;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
