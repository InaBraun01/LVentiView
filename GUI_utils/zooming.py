from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint

class ZoomableImageLabel(QLabel):
    """A QLabel that supports zooming and optional panning for images."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMinimumSize(100, 100)
        self._pixmap = None
        self._zoom = 1.0
        self._pan_active = False
        self._pan_start = QPoint(0,0)
        self._pan_offset = QPoint(0,0)
        self._is_updating = False  # ADD THIS FLAG
        
    def setPixmap(self, pixmap):
        if self._is_updating:  # Prevent re-entrant calls
            return
        self._pixmap = pixmap
        self._zoom = 1.0
        self._pan_offset = QPoint(0,0)
        if self.width() > 0 and self.height() > 0:
            super().setPixmap(self.scaled_pixmap())

    def wheelEvent(self, event):
        if self._is_updating or self._pixmap is None or self._pixmap.isNull():
            return
            
        if event.angleDelta().y() > 0:
            self._zoom *= 1.15
        else:
            self._zoom /= 1.15
        
        self._zoom = max(0.1, min(self._zoom, 10.0))
        
        if self.width() > 0 and self.height() > 0:
            super().setPixmap(self.scaled_pixmap())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_active = True
            self._pan_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_active and self._pixmap and not self._is_updating:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            if self.width() > 0 and self.height() > 0:
                super().setPixmap(self.scaled_pixmap())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_active = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        if self._pixmap and self.width() > 0 and self.height() > 0 and not self._is_updating:
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)

    def scaled_pixmap(self):
        """Return the current QPixmap scaled+transformed for zoom and pan."""
        if self._is_updating or self._pixmap is None or self._pixmap.isNull():
            return QPixmap()
        
        if self.width() <= 0 or self.height() <= 0:
            return QPixmap()
        
        self._is_updating = True  # SET FLAG
        
        try:
            base = self._pixmap
            scaled = base.scaled(
                base.size() * self._zoom, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            if scaled.isNull() or scaled.width() <= 0 or scaled.height() <= 0:
                return QPixmap()
            
            result = QPixmap(self.size())
            result.fill(Qt.transparent)
            
            painter = QPainter(result)
            x = (self.width() - scaled.width()) // 2 + self._pan_offset.x()
            y = (self.height() - scaled.height()) // 2 + self._pan_offset.y()
            painter.drawPixmap(x, y, scaled)
            painter.end()
            
            return result
        finally:
            self._is_updating = False  # CLEAR FLAG

    def clear(self):
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = QPoint(0,0)
        super().clear()
        self.setText("No image selected")