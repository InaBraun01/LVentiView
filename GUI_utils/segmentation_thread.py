import os
from PyQt5.QtCore import QThread, pyqtSignal
from Python_Code.DicomExam import DicomExam
from Python_Code.Utilis.analysis_utils import compute_cardiac_parameters
from Python_Code.Utilis.clean_MRI_utils import estimateValvePlanePosition
from Python_Code.Segmentation import segment

class AnalysisThread(QThread):
    """
    Worker thread for running the full MRI analysis pipeline.
    Handles segmentation, cleaning, landmark estimation,
    and cardiac parameter computation.
    Emits signals to update the GUI with progress and results.
    """
    log_signal = pyqtSignal(str)         # Log messages for status updates
    seg_images_signal = pyqtSignal(str)  # Path to segmentation results
    cardiac_plots_signal = pyqtSignal(str) # Path to cardiac plots
    finished_signal = pyqtSignal()       # Emitted when analysis finishes

    def __init__(self, input_folder, output_folder, do_clean=True, do_cardiac=True, clean_params=None, crop_params=None):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.do_clean = do_clean
        self.do_cardiac = do_cardiac
        self.clean_params = clean_params or {}
        self.crop_params = crop_params or {}

    def run(self):
        """
        Main analysis workflow:
        - Load DICOM data
        - Run segmentation
        - Estimate valve plane + landmarks
        - (Optional) Clean data
        - (Optional) Compute cardiac parameters
        - Save all results
        """
        try:
            os.makedirs(self.output_folder, exist_ok=True)

            # Step 1: Load exam data
            self.log_signal.emit("Loading DICOM data...")
            de = DicomExam(self.input_folder, self.output_folder)

            # Step 2: Run segmentation
            self.log_signal.emit("Running segmentation...")
            segment(de, **self.crop_params)

            # Save segmentation results
            self.log_signal.emit("Save Segmentation Results ..")
            de.save_images(prefix='full')
            seg_image_folder = os.path.join(de.folder['initial_segs'])
            if os.path.exists(seg_image_folder):
                self.seg_images_signal.emit(seg_image_folder)

            # Step 3: Landmark estimation
            self.log_signal.emit("Estimating valve plane position...")
            estimateValvePlanePosition(de)
            self.log_signal.emit("Estimating landmarks...")
            de.estimate_landmarks()

            # Step 4: Optional data cleaning
            if self.do_clean:
                self.log_signal.emit("Cleaning data...")
                de.clean_data(**self.clean_params)
                self.log_signal.emit("Saving cleaned images...")
                de.save_images(prefix='cleaned')
                if os.path.exists(seg_image_folder):
                    self.seg_images_signal.emit(seg_image_folder)
            else:
                self.log_signal.emit("Data cleaning skipped as per user selection.")

            # Step 5: Optional cardiac analysis
            if self.do_cardiac:
                self.log_signal.emit("Analyse segmentation masks...")
                compute_cardiac_parameters(de, 'seg')
                cardiac_plot_folder = os.path.join(de.folder['seg_plots']) 
                if os.path.exists(cardiac_plot_folder):
                    self.cardiac_plots_signal.emit(cardiac_plot_folder)
            else:
                self.log_signal.emit("Skipping cardiac parameter plot generation as per user selection.")

            # Step 6: Save full analysis object
            self.log_signal.emit("Saving analysis object...")
            de.save()
            self.log_signal.emit("Analysis finished.")

        except Exception as e:
            # Catch and forward any error message to the GUI
            self.log_signal.emit(f"Error during analysis: {e}")

        # Always emit finished signal so GUI can react
        self.finished_signal.emit()

