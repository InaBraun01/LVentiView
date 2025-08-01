from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from .start_page import StartPage
from .dicom_analysis_app import DicomAnalysisApp
from .mesh_generation_app import MeshGenerationApp
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# Shared data model to manage folder paths
class FolderManager(QObject):
    inputFolderChanged = pyqtSignal(str)
    outputFolderChanged = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._input_folder = ""
        self._output_folder = ""
    
    def set_input_folder(self, folder):
        if self._input_folder != folder:
            self._input_folder = folder
            self.inputFolderChanged.emit(folder)
    
    def set_output_folder(self, folder):
        if self._output_folder != folder:
            self._output_folder = folder
            self.outputFolderChanged.emit(folder)
    
    def get_input_folder(self):
        return self._input_folder
    
    def get_output_folder(self):
        return self._output_folder
    
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.folder_manager = FolderManager()

        self.setWindowTitle("Home Page")
        self.resize(900, 860)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)
        # pages
        self.start_page = StartPage(self.show_analysis_page, self.show_mesh_page)
        self.analysis_page = DicomAnalysisApp(self.folder_manager)
        self.mesh_page = MeshGenerationApp(self.folder_manager)
        # connect signals
        self.analysis_page.backRequested.connect(self.show_start_page)
        self.analysis_page.switchToMeshRequested.connect(self.show_mesh_page)
        self.mesh_page.backRequested.connect(self.show_start_page)
        # add
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(self.mesh_page)
        self.stack.setCurrentWidget(self.start_page)

    def show_analysis_page(self):
        self.stack.setCurrentWidget(self.analysis_page)
        self.setWindowTitle("Segmentation Module")

    def show_mesh_page(self):
        self.stack.setCurrentWidget(self.mesh_page)
        self.setWindowTitle("Mesh Generation Module")

    def show_start_page(self):
        self.stack.setCurrentWidget(self.start_page)
        self.setWindowTitle("Home Page")