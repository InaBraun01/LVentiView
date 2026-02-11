from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from .start_page import StartPage
from .segmentation_app import DicomAnalysisApp
from .mesh_generation_app import MeshGenerationApp
import os

# --- Shared data model to manage folder paths ---
class FolderManager(QObject):
    inputFolderChanged = pyqtSignal(str)   # emitted when input folder changes
    outputFolderChanged = pyqtSignal(str)  # emitted when output folder changes
    
    def __init__(self):
        super().__init__()
        self._input_folder = ""
        self._output_folder = ""
    
    def set_input_folder(self, folder):
        if self._input_folder != folder:
            self._input_folder = folder
            self.inputFolderChanged.emit(folder)  # notify subscribers
    
    def set_output_folder(self, folder):
        if self._output_folder != folder:
            self._output_folder = folder
            self.outputFolderChanged.emit(folder)  # notify subscribers
    
    def get_input_folder(self):
        return self._input_folder
    
    def get_output_folder(self):
        return self._output_folder

# --- Main application window with stacked pages ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.folder_manager = FolderManager()  # shared folder paths across pages
        
        # Window properties
        self.setWindowTitle("LVentiView")
        self.resize(900, 860)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Stacked widget to switch between pages
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)
        
        # --- Pages ---
        self.start_page = StartPage(self.show_analysis_page, self.show_mesh_page)
        self.analysis_page = DicomAnalysisApp(self.folder_manager)
        self.mesh_page = MeshGenerationApp(self.folder_manager)
        
        # --- Connect navigation signals ---
        self.analysis_page.backRequested.connect(self.show_start_page)
        self.analysis_page.switchToMeshRequested.connect(self.show_mesh_page)
        self.mesh_page.backRequested.connect(self.show_start_page)
        
        # --- Add pages to stacked widget ---
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(self.mesh_page)
        
        # Show start page initially
        self.stack.setCurrentWidget(self.start_page)
    
    # --- Page navigation methods ---
    def show_analysis_page(self):
        self.stack.setCurrentWidget(self.analysis_page)
    
    def show_mesh_page(self):
        self.stack.setCurrentWidget(self.mesh_page)
    
    def show_start_page(self):
        self.stack.setCurrentWidget(self.start_page)

