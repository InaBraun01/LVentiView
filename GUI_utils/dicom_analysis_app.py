import os
import csv
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem, 
                             QSizePolicy, QCheckBox, QToolButton,QGroupBox, QTabWidget,
                             QFormLayout, QLineEdit, QScrollArea, QPlainTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QIntValidator, QValidator
from .analysis_thread import AnalysisThread
from GUI_utils.zooming import ZoomableImageLabel

class PercentageValidator(QValidator):
    """Custom validator to enforce values between 0.0 and 1.0"""
    
    def validate(self, input_str, pos):
        if not input_str:
            return QValidator.Intermediate, input_str, pos
        
        try:
            value = float(input_str)
            if 0.0 <= value <= 1.0:
                return QValidator.Acceptable, input_str, pos
            else:
                return QValidator.Invalid, input_str, pos
        except ValueError:
            # Check if it's a partial valid input (like "0." or "0.5")
            if input_str in ['0', '0.', '1', '1.'] or (input_str.count('.') == 1 and input_str.replace('.', '').isdigit()):
                return QValidator.Intermediate, input_str, pos
            return QValidator.Invalid, input_str, pos

class DicomAnalysisApp(QWidget):
    backRequested = pyqtSignal()
    switchToMeshRequested = pyqtSignal()

    def __init__(self,folder_manager):
        super().__init__()
        self.folder_manager = folder_manager
        self.resize(900, 860)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # --- Scroll area setup ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        # Add scroll area to the main window layout
        main_layout.addWidget(scroll_area)

        # All widgets now go inside this scroll layout
        self.layout = scroll_layout

        # Back button top left as a QToolButton with a left arrow
        self.back_button = QToolButton()
        self.back_button.setArrowType(Qt.LeftArrow)  # Left pointing arrow
        self.back_button.setToolTip("Back to Start Page")
        self.back_button.setFixedSize(30, 30)  # Smaller fixed size
        self.back_button.clicked.connect(self.backRequested.emit)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.back_button)
        top_layout.addStretch()
        self.layout.addLayout(top_layout)

       # Input/output folders
        self.setup_folder_ui()

        self.param_fields = {}
        self.crop_param_fields = {}

        # Only tab to select parameters
        self.set_params_group = QGroupBox("Set Parameters")
        self.set_params_group.setCheckable(True)
        self.set_params_group.setChecked(False)
        self.set_params_group.toggled.connect(self.toggle_params_visibility)
        self.layout.addWidget(self.set_params_group)
        group_layout = QVBoxLayout()
        self.set_params_group.setLayout(group_layout)
        self.set_params_tab_widget = QTabWidget()
        group_layout.addWidget(self.set_params_tab_widget)
        self.set_params_tab_widget.setVisible(False) # start hidden

        #Clean data parameter page
        clean_params_widget = QWidget()
        clean_params_layout = QFormLayout(clean_params_widget)

        clean_params = [
            ('percentage_base', 'Exclusion threshold at base', '0.3'),
            ('percentage_apex', 'Exclusion threshold at apex', '0.2'),
            ('slice_threshold', 'Exclusion threshold of missing slices', '1'),
            ('margin_factor','Multiplicative padding factor', '2.5')
        ]

        for key, label, default in clean_params:
            input_widget = QLineEdit()
            input_widget.setText(default)
            
            # Set validators based on parameter type
            if key in ['percentage_base', 'percentage_apex']:
                # Use custom validator for percentage values (0.0 to 1.0)
                validator = PercentageValidator()
                input_widget.setValidator(validator)
            elif key == 'slice_threshold':
                # Integer validator for slice_threshold
                input_widget.setValidator(QIntValidator(0, 1000000, self))

            if key == 'margin_factor':
                self.crop_param_fields[key] = input_widget
            
            else:
                self.param_fields[key] = input_widget

            clean_params_layout.addRow(label + ':', input_widget)
        self.set_params_tab_widget.addTab(clean_params_widget, "MRI Cleaning Parameters")

        # Checkboxes
        self.clean_data_checkbox = QCheckBox("Data Cleaning")
        self.clean_data_checkbox.setChecked(True)
        self.layout.addWidget(self.clean_data_checkbox)

        self.cardiac_plots_checkbox = QCheckBox("Calculate Volumes")
        self.cardiac_plots_checkbox.setChecked(True)
        self.cardiac_plots_checkbox.stateChanged.connect(self.toggle_cardiac_plots_visibility)
        self.layout.addWidget(self.cardiac_plots_checkbox)

        # Run button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.btn_run)

        # Connect to folder manager signals
        self.folder_manager.inputFolderChanged.connect(self.update_input_label)
        self.folder_manager.outputFolderChanged.connect(self.update_output_label)
        
        # Initialize labels with current folder values
        self.update_input_label(self.folder_manager.get_input_folder())
        self.update_output_label(self.folder_manager.get_output_folder())

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output, stretch=1)

        # Segmentation images list
        self.seg_image_list = QListWidget()
        self.seg_image_list.itemChanged.connect(self.on_seg_image_checked)
        self.seg_image_list.setMaximumHeight(100)  # you can tune this (e.g., 80, 120)
        self.label_segmentation = QLabel("Segmentation Masks:")
        self.label_segmentation.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.label_segmentation)
        self.layout.addWidget(self.seg_image_list)

        # Cardiac plots label and list
        self.label_cardiac = QLabel("Blood Pool and Myocardium Volumes:")
        self.label_cardiac.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.cardiac_plot_list = QListWidget()
        self.cardiac_plot_list.itemChanged.connect(self.on_cardiac_plot_checked)
        self.cardiac_plot_list.setMaximumHeight(100)  # you can tune this
        self.layout.addWidget(self.label_cardiac)
        self.layout.addWidget(self.cardiac_plot_list)

        # Hide cardiac plots initially based on checkbox
        self.label_cardiac.setVisible(self.cardiac_plots_checkbox.isChecked())
        self.cardiac_plot_list.setVisible(self.cardiac_plots_checkbox.isChecked())

        # Image display
        self.image_display = ZoomableImageLabel()
        self.image_display.setText("No image selected")
        self.layout.addWidget(self.image_display, stretch=3)  # already plenty
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.image_display, stretch=3)

        #Text box to print EDV, ESV, SV and EF
        self.results_title = QLabel("Computed Cardiac Function Parameters:")
        self.results_title.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.results_title)

        # Text box to display EDV, ESV, SV and EF
        self.results_box = QPlainTextEdit()
        self.results_box.setReadOnly(True)  # Make it non-editable
        self.layout.addWidget(self.results_box)

        # Button to go to Mesh Generation
        self.to_mesh_button = QPushButton("Go to Mesh Generation Module")
        self.to_mesh_button.clicked.connect(self.switchToMeshRequested.emit)
        self.to_mesh_button.hide() 
        self.layout.addStretch()
        self.layout.addWidget(self.to_mesh_button)

        self.selected_input_folder = None
        self.selected_output_folder = None
        self.analysis_thread = None
        self.seg_image_folder = None
        self.cardiac_plot_folder = None

    def toggle_cardiac_plots_visibility(self, state):
        visible = (state == Qt.Checked)
        self.label_cardiac.setVisible(visible)
        self.cardiac_plot_list.setVisible(visible)

    def log(self, message):
        self.log_output.append(message)

    def setup_folder_ui(self):
        # Input folder
        input_layout = QHBoxLayout()
        self.input_label = QLabel("No folder selected")
        btn_pick_input = QPushButton("Select MRI Data Folder...")
        btn_pick_input.clicked.connect(self.pick_input_folder)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(btn_pick_input)
        self.layout.addLayout(input_layout)
        
        # Output folder
        output_layout = QHBoxLayout()
        self.output_label = QLabel("No folder selected")
        btn_pick_output = QPushButton("Select Output Folder")
        btn_pick_output.clicked.connect(self.pick_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(btn_pick_output)
        self.layout.addLayout(output_layout)
    
    def pick_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select MRI Data Folder")
        if folder:
            self.folder_manager.set_input_folder(folder)
    
    def pick_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_manager.set_output_folder(folder)
    
    def update_input_label(self, folder):
        self.selected_input_folder = folder 
        if folder:
            self.input_label.setText(f"Input: {folder}")
        else:
            self.input_label.setText("No folder selected")
        self.check_enable_run()
    
    def update_output_label(self, folder):
        self.selected_output_folder = folder
        if folder:
            self.output_label.setText(f"Output: {folder}")
        else:
            self.output_label.setText("No folder selected")
        self.check_enable_run()

    def check_enable_run(self):
        self.btn_run.setEnabled(bool(self.selected_input_folder and self.selected_output_folder))

    def run_analysis(self):
        if not self.selected_input_folder or not self.selected_output_folder:
            self.log("Please select both input and output folder.")
            return

        self.btn_run.setEnabled(False)
        self.log("Starting analysis...")
        clean_params = self.get_clean_params()
        crop_params = self.get_crop_params()

        self.seg_image_list.clear()
        self.cardiac_plot_list.clear()
        self.image_display.clear()
        self.image_display.setText("No image selected")

        do_clean = self.clean_data_checkbox.isChecked()
        do_cardiac = self.cardiac_plots_checkbox.isChecked()

        self.analysis_thread = AnalysisThread(
            self.selected_input_folder,
            self.selected_output_folder,
            do_clean=do_clean,
            do_cardiac=do_cardiac,
            clean_params = clean_params,
            crop_params = crop_params
        )
        self.analysis_thread.log_signal.connect(self.log)
        self.analysis_thread.seg_images_signal.connect(self.load_seg_images_from_folder)
        self.analysis_thread.cardiac_plots_signal.connect(self.load_cardiac_plots_from_folder)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.start()

    def load_seg_images_from_folder(self, folder_path):
        self.seg_image_folder = folder_path
        self.seg_image_list.blockSignals(True)
        self.seg_image_list.clear()
        try:
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            images.sort()
            for img_name in images:
                item = QListWidgetItem(img_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.seg_image_list.addItem(item)
            if not images:
                self.log(f"No segmentation images found.")
        except Exception as e:
            self.log(f"Failed to load segmentation images: {e}")
        self.seg_image_list.blockSignals(False)

    def load_cardiac_plots_from_folder(self, folder_path):
        self.cardiac_plot_folder = folder_path
        self.cardiac_plot_list.blockSignals(True)
        self.cardiac_plot_list.clear()
        try:
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            images.sort()
            for img_name in images:
                item = QListWidgetItem(img_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.cardiac_plot_list.addItem(item)

            if not images:
                self.log(f"No volume plots found.")

            # Now also load and display the metrics from CSV
            dir_name = os.path.dirname(folder_path)
            csv_path = os.path.join(dir_name, "ED_ES_state.csv")  # Need to still replace this, so that it works
            print(csv_path)
            self.load_and_display_metrics(csv_path)

        except Exception as e:
            self.log(f"Failed to load volume plots: {e}")
        self.cardiac_plot_list.blockSignals(False)

    def on_seg_image_checked(self, item):
        if item.checkState() == Qt.Checked:
            self.block_cardiac_list_signals(True)
            for i in range(self.cardiac_plot_list.count()):
                self.cardiac_plot_list.item(i).setCheckState(Qt.Unchecked)
            self.block_cardiac_list_signals(False)

            for i in range(self.seg_image_list.count()):
                other = self.seg_image_list.item(i)
                if other is not item and other.checkState() == Qt.Checked:
                    other.setCheckState(Qt.Unchecked)

            self.display_image(self.seg_image_folder, item.text())
        else:
            if not any(self.seg_image_list.item(i).checkState() == Qt.Checked for i in range(self.seg_image_list.count())):
                self.image_display.clear()
                self.image_display.setText("No image selected")

    def on_cardiac_plot_checked(self, item):
        if item.checkState() == Qt.Checked:
            self.block_seg_list_signals(True)
            for i in range(self.seg_image_list.count()):
                self.seg_image_list.item(i).setCheckState(Qt.Unchecked)
            self.block_seg_list_signals(False)

            for i in range(self.cardiac_plot_list.count()):
                other = self.cardiac_plot_list.item(i)
                if other is not item and other.checkState() == Qt.Checked:
                    other.setCheckState(Qt.Unchecked)

            self.display_image(self.cardiac_plot_folder, item.text())
        else:
            if not any(self.cardiac_plot_list.item(i).checkState() == Qt.Checked for i in range(self.cardiac_plot_list.count())):
                self.image_display.clear()
                self.image_display.setText("No image selected")

    def block_seg_list_signals(self, block):
        self.seg_image_list.blockSignals(block)

    def block_cardiac_list_signals(self, block):
        self.cardiac_plot_list.blockSignals(block)

    def display_image(self, folder, image_name):
        if not folder:
            return
        image_path = os.path.join(folder, image_name)
        if not os.path.exists(image_path):
            self.log(f"Image does not exist: {image_path}")
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.log(f"Failed to load image: {image_path}")
            self.image_display.setText("Failed to load image")
            return
        scaled_pix = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pix)

    def analysis_finished(self):
        self.btn_run.setEnabled(True)
        self.to_mesh_button.show()

    def toggle_params_visibility(self, checked):
        self.set_params_tab_widget.setVisible(checked)
    

    def get_clean_params(self):
        params = {}
        for key, widget in self.param_fields.items():
            text = widget.text().strip()

            if key in ['percentage_base', 'percentage_apex']:
                try:
                    value = float(text)
                    if 0 <= value <= 1:
                        params[key] = value
                    else:
                        self.log_output.append(f"Value for {key} must be between 0 and 1: {text}")
                        params[key] = 0.0  # Default fallback
                except ValueError:
                    self.log_output.append(f"Invalid float for {key}: {text}")
                    params[key] = 0.0  # Default fallback
            elif key == 'slice_threshold':
                try:
                    params[key] = int(text)
                except ValueError:
                    self.log_output.append(f"Invalid integer for {key}: {text}")
        return params
    

    def get_crop_params(self):
        params = {}
        for key, widget in self.crop_param_fields.items():
            text = widget.text().strip()
            value = float(text)
            params[key] = value

        return params
    
    import csv

    def load_and_display_metrics(self, csv_path):
        if not os.path.exists(csv_path):
            self.results_box.setPlainText("Results file not found.")
            return

        lines = []

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['Parameter']
                value = row['Value']
                time_step = row['Time_step']
                if name=="EF":
                    lines.append(f"{name}: {value}%")
                elif name=="SV":
                    lines.append(f"{name}: {value}ml")
                else:
                    lines.append(f"{name} =  {value}ml. Calculated for time step: {time_step}")

        result_text = "\n".join(lines)
        self.results_box.setPlainText(result_text)
