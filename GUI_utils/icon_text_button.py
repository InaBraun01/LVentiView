from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QFont, QCursor
from PyQt5.QtCore import Qt, pyqtSignal

class IconTextButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, icon_path, text, size=150, icon_ratio=0.6, parent=None):
        super().__init__(parent)
        self.size = size
        self.icon_ratio = icon_ratio
        self.setFixedSize(self.size, self.size)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setLayout(layout)
        self.icon_label = QLabel()
        icon_pixmap = QPixmap(icon_path).scaled(
            self.size, int(self.size * self.icon_ratio),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(icon_pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.icon_label, stretch=int(self.icon_ratio * 1000))
        self.text_label = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        self.text_label.setFont(font)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.text_label, stretch=int((1 - self.icon_ratio) * 1000))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            event.ignore()