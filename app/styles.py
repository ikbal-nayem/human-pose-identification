STYLE_SHEET = """
* {
    font-family: 'Segoe UI', sans-serif;
    outline: none;
}

/* ---- window shell ------------------------------------------------------ */
QWidget#windowFrame {
    background-color: transparent;
    border: 1px solid #2b2d3a;
    border-radius: 8px;
}

QWidget#resizeGrip {
    background-color: transparent;
}

QWidget#titleBar {
    background-color: #131319;
    border-bottom: 1px solid #262835;
}

QLabel#titleBarText {
    color: #9aa0b8;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

QPushButton#titleBarButton, QPushButton#titleBarCloseButton {
    background: transparent;
    border: none;
    border-radius: 0px;
    color: #9aa0b8;
    font-size: 13px;
    padding: 0px;
}

QPushButton#titleBarButton:hover {
    background-color: #24263a;
    color: #e7e9f2;
}

QPushButton#titleBarCloseButton:hover {
    background-color: #d9455f;
    color: #ffffff;
}

QMainWindow, QWidget#root {
    background-color: #1c1d25;
}

QWidget {
    color: #dcdfeb;
}

/* ---- sidebar --------------------------------------------------------- */
QWidget#sidebar {
    background-color: #15161c;
    border-right: 1px solid #24263a;
}

QLabel#sidebarSection {
    color: #666c82;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 20px 18px 10px 18px;
}

QLabel#appTitle {
    color: #eef0f7;
    font-size: 17px;
    font-weight: 600;
    padding: 22px 18px 4px 18px;
}

QPushButton#navButton {
    text-align: left;
    padding: 12px 18px;
    border: none;
    border-radius: 0px;
    background: transparent;
    color: #a6abc0;
    font-size: 13px;
    font-weight: 500;
}

QPushButton#navButton:hover {
    background-color: #1b1d27;
    color: #eef0f7;
}

QPushButton#navButton:checked {
    background-color: #201f39;
    color: #a596ff;
    border-left: 3px solid #7c6cf0;
}

/* ---- generic surfaces ------------------------------------------------- */
QWidget#page {
    background-color: #1c1d25;
}

QFrame#card {
    background-color: #212230;
    border: 1px solid #2b2d3c;
    border-radius: 12px;
}

QLabel#pageTitle {
    font-size: 20px;
    font-weight: 600;
    color: #eef0f7;
}

QLabel#pageHint {
    color: #868ca3;
    font-size: 12px;
}

QLabel#sectionLabel {
    color: #969cb4;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}

/* ---- buttons ------------------------------------------------------------ */
QPushButton {
    background-color: #282a3a;
    color: #dcdfeb;
    border: 1px solid #34364a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #313349;
    border-color: #454868;
}

QPushButton:pressed {
    background-color: #232432;
}

QPushButton:disabled {
    color: #565a6c;
    background-color: #1e2029;
    border-color: #262835;
}

QPushButton#primaryButton {
    background-color: #7c6cf0;
    border: 1px solid #7c6cf0;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #8f80ff;
}

QPushButton#dangerButton {
    background-color: transparent;
    border: 1px solid #4a2b34;
    color: #ef8b9c;
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
    background-color: #ef5d78;
    border: 1px solid #ef5d78;
    color: #2a0810;
    font-weight: 700;
    font-size: 14px;
    padding: 12px 22px;
    border-radius: 10px;
}

QPushButton#stopButton:hover {
    background-color: #ff7690;
}

QPushButton#keyCapture {
    background-color: #262838;
    border: 1px solid #3b3e58;
    border-radius: 6px;
    color: #767c92;
    font-style: italic;
    padding: 5px 10px;
    min-width: 90px;
}

QPushButton#keyCapture[hasKey="true"] {
    color: #eef0f7;
    font-weight: 700;
    font-style: normal;
    letter-spacing: 0.5px;
    border-color: #4a4d6a;
    background-color: #2b2d42;
}

QPushButton#keyCapture:hover {
    border-color: #7c6cf0;
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
    color: #ef8b9c;
}

/* ---- inputs -------------------------------------------------------------- */
QLineEdit, QComboBox {
    background-color: #1e2029;
    border: 1px solid #34364a;
    border-radius: 8px;
    padding: 7px 10px;
    color: #dcdfeb;
    font-size: 12px;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #7c6cf0;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #20222d;
    border: 1px solid #34364a;
    selection-background-color: #2c2a4d;
    color: #dcdfeb;
    outline: none;
}

/* ---- lists / tables -------------------------------------------------------- */
QListWidget {
    background-color: #1e2029;
    border: 1px solid #2b2d3c;
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
    background-color: #2c2a4d;
    color: #eef0f7;
}

QListWidget::item:hover:!selected {
    background-color: #262838;
}

QTableWidget {
    background-color: #1e2029;
    border: 1px solid #2b2d3c;
    border-radius: 10px;
    gridline-color: #262835;
    font-size: 12px;
    selection-background-color: #262838;
}

QHeaderView::section {
    background-color: #1a1c25;
    color: #969cb4;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #2b2d3c;
    font-size: 11px;
    font-weight: 600;
}

QTableWidget::item {
    padding: 4px;
}


QLabel#videoLabel {
    background-color: #000000;
    border: 1px solid #2b2d3c;
    border-radius: 12px;
}

QLabel#activityChip {
    background-color: #204536;
    border: 1px solid #2f6b52;
    border-radius: 12px;
    padding: 4px 12px;
    color: #7bd9ad;
    font-size: 11px;
    font-weight: 600;
}

QLabel#statusBadge {
    background-color: #262838;
    border: 1px solid #3b3e58;
    border-radius: 12px;
    padding: 4px 10px;
    color: #a596ff;
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
