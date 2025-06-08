from PyQt5.QtCore import QThread, pyqtSignal

class AnalysisStepThread(QThread):
    log_signal = pyqtSignal(str)
    plot_folder_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, func, de):
        super().__init__()
        self.func = func
        self.de = de

    
    def run(self):
        try:
            folder = self.func(self.de, log_func=self.log_signal.emit)
            self.plot_folder_signal.emit(folder)
        except Exception as e:
            self.log_signal.emit(f"Analysis step failed: {e}")
        finally:
            # Always emit finished signal, whether successful or not
            self.finished_signal.emit()