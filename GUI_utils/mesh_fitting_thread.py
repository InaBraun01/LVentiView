from PyQt5.QtCore import QThread, pyqtSignal
from .mesh_generation import mesh_fit_save_images

class MeshGenerationThread(QThread):
    # Signals for GUI updates
    log_signal = pyqtSignal(str)               # text log messages
    progress_signal = pyqtSignal(int, int)     # progress updates (current, total)
    image_folder_signal = pyqtSignal(str)      # folder containing generated mesh images
    finished_signal = pyqtSignal()             # emitted when thread finishes

    def __init__(self, input_path, output_folder, fit_params=None):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder
        self.fit_params = fit_params or {}      # parameters for mesh fitting
        self.de = None                          # will store the loaded DicomExam object

    def run(self):
        """Run mesh fitting in a separate thread to avoid blocking GUI."""
        try:
            # Perform mesh fitting and save mesh images
            image_folder, de = mesh_fit_save_images(
                self.input_path,
                self.output_folder,
                log_func=self.log_signal.emit,       # send log messages to GUI
                progress_func=self.progress_signal.emit,  # send progress updates
                **self.fit_params
            )
            self.de = de
            self.image_folder_signal.emit(image_folder)  # notify GUI of folder location
        except Exception as e:
            self.log_signal.emit(f"Mesh generation failed: {e}")  # report errors
        self.finished_signal.emit()  # always signal that thread has finished
