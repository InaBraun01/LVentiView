from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint

class ZoomableImageLabel(QLabel):
    """A QLabel that supports zooming and optional panning for images."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)                  # Center image by default
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMinimumSize(100, 100)
        self._pixmap = None           # Store the raw current pixmap
        self._zoom = 1.0              # Current zoom factor
        self._pan_active = False      # Whether right-click panning is active
        self._pan_start = QPoint(0,0) # Mouse position where panning started
        self._pan_offset = QPoint(0,0)# Current offset due to panning
        
    def setPixmap(self, pixmap):
        # Reset zoom and pan when a new pixmap is set
        self._pixmap = pixmap
        self._zoom = 1.0
        self._pan_offset = QPoint(0,0)
        super().setPixmap(self.scaled_pixmap())
        
    def wheelEvent(self, event):
        # Zoom in/out with mouse wheel
        if event.angleDelta().y() > 0:
            self._zoom *= 1.15
        else:
            self._zoom /= 1.15
        # Clamp zoom level between 0.1x and 10x
        self._zoom = max(0.1, min(self._zoom, 10.0))
        if self._pixmap:
            super().setPixmap(self.scaled_pixmap())
    
    def mousePressEvent(self, event):
        # Start panning on right mouse button press
        if event.button() == Qt.RightButton:
            self._pan_active = True
            self._pan_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Apply pan offset while dragging with right mouse button
        if self._pan_active and self._pixmap:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            super().setPixmap(self.scaled_pixmap())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        # Stop panning on right mouse button release
        if event.button() == Qt.RightButton:
            self._pan_active = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        # Rescale pixmap when widget is resized
        if self._pixmap:
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)

    def scaled_pixmap(self):
        """Return the current QPixmap scaled+transformed for zoom and pan."""
        if self._pixmap is None: 
            return QPixmap()
        base = self._pixmap
        # Apply zoom scaling
        scaled = base.scaled(base.size() * self._zoom, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Create result pixmap with widget size, so image is centered (and panned)
        result = QPixmap(self.size())
        result.fill(Qt.transparent)
        painter = QPainter(result)
        # Compute placement: center image, then apply pan offset
        x = (self.width() - scaled.width()) // 2 + self._pan_offset.x()
        y = (self.height() - scaled.height()) // 2 + self._pan_offset.y()
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return result

    def clear(self):
        # Reset state and show placeholder text
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = QPoint(0,0)
        super().clear()
        self.setText("No image selected")
