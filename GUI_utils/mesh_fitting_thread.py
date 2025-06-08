from PyQt5.QtCore import QThread, pyqtSignal
from .mesh_generation import mesh_fit_save_images

class MeshGenerationThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  
    image_folder_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_path, output_folder, fit_params=None):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder
        self.fit_params = fit_params or {}
        self.de = None

    def run(self):
        try:
            image_folder, de = mesh_fit_save_images(
                self.input_path,
                self.output_folder,
                log_func=self.log_signal.emit,
                progress_func=self.progress_signal.emit,
                **self.fit_params
            )
            self.de = de
            self.image_folder_signal.emit(image_folder)
        except Exception as e:
            self.log_signal.emit(f"Mesh generation failed: {e}")
        self.finished_signal.emit()