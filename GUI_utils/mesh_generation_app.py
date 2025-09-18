import os
import sys
import glob
import csv
import statistics
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit,
    QToolButton, QLineEdit, QGroupBox, QFormLayout, QTabWidget, QComboBox, QListWidget, 
    QListWidgetItem, QCheckBox, QProgressBar, QScrollArea, QPlainTextEdit
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIntValidator, QPixmap

from .mesh_generation import (
    analyze_mesh_volumes_step,
    calculate_segmentation_thickness_step
)
from .analysis_step_thread import AnalysisStepThread
from .mesh_fitting_thread import MeshGenerationThread

class MeshGenerationApp(QWidget):
    backRequested = pyqtSignal()

    def __init__(self,folder_manager):
        super().__init__()

        self.folder_manager = folder_manager
        self.resize(900, 900)

         # === Outer layout for the entire app ===
        outer_layout = QVBoxLayout(self)

        # === Scroll Area setup ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # important for resizing
        outer_layout.addWidget(scroll_area)

        # === Inner widget that goes inside the scroll area ===
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        # === Main content layout inside the scroll area ===
        self.layout = QVBoxLayout(scroll_widget)  # note: self.layout is now on scroll_widget

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

        # Only tab widget goes into parameter group!
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

        # Mesh fitting parameters page
        fitting_params_widget = QWidget()
        fitting_params_layout = QFormLayout(fitting_params_widget)

        fitting_params = [
            ('time_frames_to_fit', 'Time Frames to be Fit', 'all'),
            ('training_steps', 'Training Steps', '1'),
            ('lr', 'Learning Rate', '0.003'),
            ('num_modes', 'Number of Modes', '25'),
        ]

        for key, label, default in fitting_params:
            if key == 'time_frames_to_fit':
                input_widget = QComboBox()
                input_widget.setEditable(True)  # Allow custom input
                input_widget.addItems(['all'])
                # Set default value
                if default in ['all']:
                    index = input_widget.findText(default)
                    if index >= 0:
                        input_widget.setCurrentIndex(index)
                else:
                    input_widget.setCurrentText(default)
                # Add placeholder text to help user understand the format
                input_widget.setEditText(default)
                input_widget.lineEdit().setPlaceholderText("Enter 'all or comma-separated positive integers (e.g., 1,2,3 for frames 1,2,3)")
            else:
                input_widget = QLineEdit()
                input_widget.setText(default)
                if key in ['num_modes', 'training_steps']:
                    input_widget.setValidator(QIntValidator(0, 1000000, self))
            
            fitting_params_layout.addRow(label + ':', input_widget)
            self.param_fields[key] = input_widget

        self.set_params_tab_widget.addTab(fitting_params_widget, "Mesh Fitting Parameters")


        # Advanced fitting parameters page
        advanced_fitting_params_widget = QWidget()
        advanced_fitting_layout = QFormLayout(advanced_fitting_params_widget)

        advanced_fitting_parms = [
            ('cp_frequency', 'Control Point Frequency', '50'),
            ('mesh_model_dir', 'Shape Model Directory', 'ShapeModel'),
            ('steps_between_fig_saves', 'Steps between Mesh Update', '50')
        ]

        for key, label, default in advanced_fitting_parms:
            input_widget = QLineEdit()
            input_widget.setText(default)
            # Fixed the condition - was checking if key equals a list, now checks if key is in the list
            if key in [ 'cp_frequency','steps_between_fig_saves']:
                input_widget.setValidator(QIntValidator(0, 1000000, self))
            advanced_fitting_layout.addRow(label + ':', input_widget)
            self.param_fields[key] = input_widget

        self.set_params_tab_widget.addTab(advanced_fitting_params_widget, "Advanced Fitting Parameters")

        # Shift and Rotation parameters page
        shift_rotation_params_widget = QWidget()
        shift_rotation_layout = QFormLayout(shift_rotation_params_widget)

        shift_rotation_params = [
            ('allow_global_shift_xy', 'Global x,y Shifts', 'True'),
            ('allow_global_shift_z', 'Global z Shifts', 'True'),
            ('allow_slice_shift', 'Slice Shifts', 'True'),
            ('allow_rotations', 'Global Rotations', 'True')
        ]

        for key, label, default in shift_rotation_params:
            input_widget = QComboBox()
            input_widget.addItems(['True', 'False'])
            index = input_widget.findText(default)
            if index >= 0:
                input_widget.setCurrentIndex(index)
            shift_rotation_layout.addRow(label + ':', input_widget)
            self.param_fields[key] = input_widget

        self.set_params_tab_widget.addTab(shift_rotation_params_widget, "Regulate Shifts + Rotations")

        #Parameters for weghts in loss function
        weight_params_widget = QWidget()
        weight_layout = QFormLayout(weight_params_widget)

        weight_params = [
            ('dice_loss_weight', "Dice loss weight", '5'),
            ('mode_loss_weight', 'Mode weight', '0.05'),
            ('global_shift_penalty_weigth', 'Global Shift Penalty', '0.3'),
            ('slice_shift_penalty_weigth', 'Slice Shift Penalty', '10'),
            ('rotation_penalty_weigth', 'Rotation Penalty', '1')
        ]

        for key, label, default in weight_params:
            input_widget = QLineEdit()
            input_widget.setText(default)
            weight_layout.addRow(label + ':', input_widget)
            self.param_fields[key] = input_widget

        self.set_params_tab_widget.addTab(weight_params_widget, "Loss Function Weights")

        self.volumetric_mesh_checkbox = QCheckBox("Calculate Volumes")
        self.volumetric_mesh_checkbox.setChecked(True)
        self.volumetric_mesh_checkbox.stateChanged.connect(self.on_volumetric_mesh_checked)
        self.layout.addWidget(self.volumetric_mesh_checkbox)

        self.thickness_checkbox = QCheckBox("Calculate Thickness")
        self.thickness_checkbox.setChecked(True)
        self.thickness_checkbox.stateChanged.connect(self.on_thickness_checked)
        self.layout.addWidget(self.thickness_checkbox)

        # --- ALL RESULTS WIDGETS OUTSIDE THE PARAM GROUP ---
        self.run_button = QPushButton("Run Mesh Generation")
        self.run_button.clicked.connect(self.run_mesh_generation)
        self.run_button.setEnabled(False)
        self.layout.addWidget(self.run_button)

        # Connect to folder manager signals
        self.folder_manager.inputFolderChanged.connect(self.update_input_label)
        self.folder_manager.outputFolderChanged.connect(self.update_output_label)

        # Initialize labels with current folder values
        self.update_input_label(self.folder_manager.get_input_folder())
        self.update_output_label(self.folder_manager.get_output_folder())

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output, stretch=1)

        # Add progress bar
        self.progressBar = QProgressBar()
        self.layout.addWidget(self.progressBar)

        # Mesh visualization list
        self.mesh_image_list = QListWidget()
        self.mesh_image_list.setMaximumHeight(40)
        self.mesh_image_list.itemChanged.connect(self.on_image_checked)
        self.label_mesh_image = QLabel("Mesh Visualization:")
        self.label_mesh_image.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.label_mesh_image)
        self.layout.addWidget(self.mesh_image_list)

        #Text box to print Blood pool and Myocardial dice
        self.dice_title = QLabel("Fit of Meshes to Segmentation Masks:")
        self.dice_title.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.dice_title)
        self.dice_box = QPlainTextEdit()
        self.dice_box.setReadOnly(True)  # Make it non-editable
        self.dice_box.setMaximumHeight(40)
        self.layout.addWidget(self.dice_box)

        # Two separate analysis plot lists with labels
        #For volume calculation
        self.volumetric_analysis_list = QListWidget()
        self.volumetric_analysis_list.setMaximumHeight(40)
        self.volumetric_analysis_list.itemChanged.connect(self.on_image_checked)
        self.volumetric_analysis_label= QLabel("Blood Pool and Myocardium Volumes:")
        self.volumetric_analysis_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.volumetric_analysis_label)
        self.layout.addWidget(self.volumetric_analysis_list)

        #for thickness calculation
        self.thickness_analysis_list = QListWidget()
        self.thickness_analysis_list.setMaximumHeight(200)
        self.thickness_analysis_list.itemChanged.connect(self.on_image_checked)
        self.thickness_analysis_label = QLabel("Local Thickness Estimation:")
        self.thickness_analysis_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.thickness_analysis_label)
        self.layout.addWidget(self.thickness_analysis_list)


        self.image_display = QLabel("No image selected")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(300)
        self.layout.addWidget(self.image_display)

        #Text box to print EDV, ESV, SV and EF
        self.results_title = QLabel("Computed Cardiac Function Parameters:")
        self.results_title.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
        self.layout.addWidget(self.results_title)
        self.results_box = QPlainTextEdit()
        self.results_box.setReadOnly(True)  # Make it non-editable
        self.layout.addWidget(self.results_box)

        self.volumetric_mesh_checked = self.volumetric_mesh_checkbox.isChecked()
        self.thickness_checked = self.thickness_checkbox.isChecked()
        self.pending_analysis_steps = []

        self.input_path = None
        self.output_folder = None
        self.mesh_thread = None
        self.analysis_step_thread = None
        self.buttons_created = False

        # Keep track of current analysis step type
        self.current_analysis_step = None
        
        # Set initial visibility based on checkbox states
        self.update_analysis_widgets_visibility()

    def toggle_params_visibility(self, checked):
        self.set_params_tab_widget.setVisible(checked)

    def setup_folder_ui(self):
        # Input folder
        input_layout = QHBoxLayout()
        self.input_label = QLabel("No folder selected")
        btn_pick_input = QPushButton("Pick segmented data folder...")
        btn_pick_input.clicked.connect(self.pick_input_folder)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(btn_pick_input)
        self.layout.addLayout(input_layout)
        
        # Output folder
        output_layout = QHBoxLayout()
        self.output_label = QLabel("No folder selected")
        btn_pick_output = QPushButton("Pick Output Folder")
        btn_pick_output.clicked.connect(self.pick_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(btn_pick_output)
        self.layout.addLayout(output_layout)
        
    def pick_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Segmented Data Folder")
        if folder:
            self.folder_manager.set_input_folder(folder)

    def pick_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_manager.set_output_folder(folder)
    
    def update_input_label(self, folder):
        self.input_path = folder
        if folder:
            self.input_label.setText(f"Input: {folder}")
        else:
            self.input_label.setText("No folder selected")
        self.check_enable_run()
    
    def update_output_label(self, folder):
        self.output_folder = folder
        if folder:
            self.output_label.setText(f"Output: {folder}")
        else:
            self.output_label.setText("No folder selected")
        self.check_enable_run()

    def check_enable_run(self):
        self.run_button.setEnabled(bool(self.input_path and self.output_folder))
        
    def get_fit_params(self):
        """Get parameters with proper type conversion"""
        params = {}
        # Define parameter types for proper conversion
        integer_params = [
            'num_cycle', 'num_modes', 'training_steps', 'steps_between_progress_update',
            'burn_in_length', 'cp_frequency', 'steps_between_fig_saves'
        ]
        float_params = [
            'lr', 'dice_loss_weight', 'mode_loss_weight', 'global_shift_penalty_weigth',
            'slice_shift_penalty_weigth', 'rotation_penalty_weigth'
        ]
        boolean_params = [
            'show_progress', 'random_starting_mesh', 'allow_global_shift_xy',
            'allow_global_shift_z', 'allow_slice_shift', 'allow_rotations'
        ]
        # Special parameters that need custom parsing
        special_params = ['time_frames_to_fit']
        
        for key, widget in self.param_fields.items():
            if isinstance(widget, QComboBox):
                text = widget.currentText()
            else:
                text = widget.text().strip()
            
            try:
                if key in integer_params:
                    params[key] = int(text) if text else 0
                elif key in float_params:
                    params[key] = float(text) if text else 0.0
                elif key in boolean_params:
                    params[key] = text.lower() in ('true', '1', 'yes')
                elif key in special_params:
                    if key == 'time_frames_to_fit':
                        # Parse time_frames_to_fit parameter
                        if text in ['all', 'all_loop']:
                            params[key] = text
                        else:
                            # Try to parse as comma-separated integers
                            try:
                                frame_list = [int(x.strip()) for x in text.split(',') if x.strip()]
                                if not frame_list:  # Empty list
                                    raise ValueError("Empty integer list")
                                # Validate that all integers are greater than 0
                                if any(frame <= 0 for frame in frame_list):
                                    raise ValueError("All frame numbers must be greater than 0")
                                # Convert from 1-based to 0-based indexing for internal use
                                params[key] = [frame - 1 for frame in frame_list]
                            except ValueError as ve:
                                if "invalid literal" in str(ve):
                                    raise ValueError(f"time_frames_to_fit must be 'all', 'all_loop', or comma-separated positive integers")
                                else:
                                    raise ve
                else:
                    # String parameters
                    params[key] = text
                    
            except ValueError as e:
                error_msg = f"Invalid value for {key}: '{text}'. Using default."
                self.log_output.append(error_msg)
                # Set default values for failed conversions
                if key in integer_params:
                    if key == 'cp_frequency':
                        params[key] = 50  # Default cp_frequency
                    elif key in ['num_modes']:
                        params[key] = 25
                    elif key in ['training_steps']:
                        params[key] = 1
                    else:
                        params[key] = 1
                elif key in float_params:
                    if key == 'lr':
                        params[key] = 0.003
                    else:
                        params[key] = 0.0
                elif key in boolean_params:
                    params[key] = True
                elif key == 'time_frames_to_fit':
                    params[key] = 'all'  # Default fallback
                else:
                    params[key] = ""
        
        return params

    def toggle_volumetric_mesh_visibility(self, state):
        visible = (state == Qt.Checked)
        self.volumetric_mesh_checkbox.setChecked(visible)

    def toggle_thickness_visibility(self, state):
        visible = (state == Qt.Checked)
        self.thickness_checkbox.setChecked(visible)

    def run_mesh_generation(self):
        self.log_output.append("Starting mesh fitting...")
        self.log_output.append("=" * 50)
        self.run_button.setEnabled(False)
        fit_params = self.get_fit_params()
        self.log_output.append("=" * 50)
        self.mesh_thread = MeshGenerationThread(self.input_path, self.output_folder, fit_params=fit_params)
        self.mesh_thread.progress_signal.connect(self.update_progress)
        self.mesh_thread.log_signal.connect(self.log_output.append)
        self.mesh_thread.image_folder_signal.connect(self.on_mesh_visual_ready)
        self.mesh_thread.finished_signal.connect(self.on_mesh_fitting_done)
        self.mesh_thread.start()

    def on_mesh_fitting_done(self):
        self.log_output.append("Mesh fitting finished.")
        self.run_button.setEnabled(True)

        self.pending_analysis_steps.clear()

        if self.volumetric_mesh_checkbox.isChecked():
            self.pending_analysis_steps.append((analyze_mesh_volumes_step, 'volumetric'))
        if self.thickness_checkbox.isChecked():
            self.pending_analysis_steps.append((calculate_segmentation_thickness_step, 'thickness'))

        self.run_next_analysis_step()

    def on_image_checked(self, item):
        if item.checkState() == Qt.Checked:
            # Uncheck all other items in all lists
            self.block_all_list_signals(True)
            
            # Get all list widgets,
            all_lists = [self.mesh_image_list, self.volumetric_analysis_list, self.thickness_analysis_list]
            
            for list_widget in all_lists:
                for i in range(list_widget.count()):
                    other = list_widget.item(i)
                    if other is not item:
                        other.setCheckState(Qt.Unchecked)
            
            self.block_all_list_signals(False)

            # Show selected image
            img_path = item.data(Qt.UserRole)
            if img_path and os.path.exists(img_path):
                self.display_image(img_path)
            else:
                self.image_display.setText("Image not found")
        else:
            # Check if any image is still selected across all lists
            all_lists = [self.mesh_image_list ,self.volumetric_analysis_list, self.thickness_analysis_list]
            
            any_checked = False
            for list_widget in all_lists:
                for i in range(list_widget.count()):
                    if list_widget.item(i).checkState() == Qt.Checked:
                        any_checked = True
                        break
                if any_checked:
                    break
            
            if not any_checked:
                self.image_display.setText("No image selected")

    def _run_analysis_step(self, func, step_type):
        self.current_analysis_step = step_type
        self.analysis_step_thread = AnalysisStepThread(
            func, self.mesh_thread.de
        )
        self.analysis_step_thread.log_signal.connect(self.log_output.append)
        self.analysis_step_thread.plot_folder_signal.connect(self.on_analysis_plots_ready)
        self.analysis_step_thread.finished_signal.connect(self.on_analysis_done)
        self.analysis_step_thread.start()

    def on_analysis_done(self):
        self.log_output.append("Analysis step finished.")
        
        # Ensure proper thread cleanup
        if self.analysis_step_thread:
            self.analysis_step_thread.wait()  # Wait for thread to finish
            self.analysis_step_thread.deleteLater()
            self.analysis_step_thread = None
        
        self.current_analysis_step = None
        
        # Use QTimer to ensure the next step runs in the next event loop cycle
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.run_next_analysis_step)

    def on_mesh_visual_ready(self, plot_folder):
        try:
            patterns = ("*.pdf","*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
            image_files = []
            for p in patterns:
                image_files.extend(glob.glob(os.path.join(plot_folder, p)))

            for img_path in sorted(image_files):
                item = QListWidgetItem(os.path.basename(img_path))
                item.setData(Qt.UserRole, img_path)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.mesh_image_list.addItem(item)

            if self.mesh_image_list.count() > 0 and self.image_display.pixmap() is None:
                self.mesh_image_list.setCurrentRow(0)
                first_item = self.mesh_image_list.item(0)
                if first_item:
                    first_item.setCheckState(Qt.Checked)

            
            dir_name = os.path.dirname(os.path.dirname(plot_folder))
            csv_path = os.path.join(dir_name, "end_dice.csv")  # Need to still replace this, so that it works
            self.load_and_display_dice_stats(csv_path)

        except Exception as e:
            self.log_output.append(f"Error processing mesh plots: {str(e)}")

    def on_analysis_plots_ready(self, plot_folder):
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff","*.pdf")
        image_files = []
        for p in patterns:
            image_files.extend(glob.glob(os.path.join(plot_folder, p)))


        # Determine which list to add to based on current analysis step
        target_list = None
        if self.current_analysis_step == 'volumetric':
            target_list = self.volumetric_analysis_list
            # Now also load and display the metrics from CSV
            dir_name = os.path.dirname(plot_folder)
            csv_path = os.path.join(dir_name, "ED_ES_states.csv")  # Need to still replace this, so that it works
            self.load_and_display_metrics(csv_path)

        elif self.current_analysis_step == 'thickness':
            target_list = self.thickness_analysis_list
        

        if target_list is None:
            self.log_output.append(f"Warning: Unknown analysis step type: {self.current_analysis_step}")
            return

        for img_path in sorted(image_files):
            item = QListWidgetItem(os.path.basename(img_path))
            item.setData(Qt.UserRole, img_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            target_list.addItem(item)

        # Auto-select first item if no image is currently displayed
        if target_list.count() > 0 and self.image_display.pixmap() is None:
            target_list.setCurrentRow(0)
            first_item = target_list.item(0)
            if first_item:
                first_item.setCheckState(Qt.Checked)

        # except Exception as e:
        #     print(e)
        #     self.log_output.append(f"Error processing analysis plots: {str(e)}")

    def run_analyse_volumetric(self):
        self._run_analysis_step(analyze_mesh_volumes_step, 'volumetric')

    def run_analyse_thickness(self):
        self._run_analysis_step(calculate_segmentation_thickness_step, 'thickness')


    def on_volumetric_mesh_checked(self):
        self.volumetric_mesh_checked = self.volumetric_mesh_checkbox.isChecked()
        self.update_volumetric_widgets_visibility()

    def on_thickness_checked(self):
        self.thickness_checked = self.thickness_checkbox.isChecked()
        self.update_thickness_widgets_visibility()

    def update_analysis_widgets_visibility(self):
        """Update visibility of all analysis widgets based on checkbox states"""
        self.update_volumetric_widgets_visibility()
        self.update_thickness_widgets_visibility()


    def update_volumetric_widgets_visibility(self):
        """Show/hide volumetric analysis widgets based on checkbox state"""
        visible = self.volumetric_mesh_checkbox.isChecked()
        self.volumetric_analysis_label.setVisible(visible)
        self.volumetric_analysis_list.setVisible(visible)

    def update_thickness_widgets_visibility(self):
        """Show/hide thickness analysis widgets based on checkbox state"""
        visible = self.thickness_checkbox.isChecked()
        self.thickness_analysis_label.setVisible(visible)
        self.thickness_analysis_list.setVisible(visible)

    def log(self, message):
        self.log_output.append(message)

    def block_all_list_signals(self, block):
        """Block or unblock signals for all list widgets"""
        self.mesh_image_list.blockSignals(block)
        self.volumetric_analysis_list.blockSignals(block)
        self.thickness_analysis_list.blockSignals(block)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display.setPixmap(scaled)
        else:
            self.image_display.setText("Could not load image.")

    def run_next_analysis_step(self):
        if self.pending_analysis_steps:
            next_step, step_type = self.pending_analysis_steps.pop(0)
            self._run_analysis_step(next_step, step_type)
        else:
            self.log_output.append("All selected analysis steps completed.")
            self.run_button.setEnabled(True)

    def update_progress(self, current, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(current)



    def load_and_display_metrics(self, csv_path):
        if not os.path.exists(csv_path):
            self.results_box.setPlainText("Results file not found.")
            return

        lines = []

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['state']
                value_str = row['volume']
                time_step = row['time_frame']

                # Try converting to float, fallback to 0.0 if conversion fails
                try:
                    value = float(value_str)
                except ValueError:
                    value = 0.0

                if name == "EF":
                    lines.append(f"{name}: {value:.2f}%")
                elif name == "SV":
                    lines.append(f"{name}: {value:.2f} ml")
                else:
                    lines.append(f"{name} = {value:.2f} ml. Calculated for time step: {time_step}")

        result_text = "\n".join(lines)
        self.results_box.setPlainText(result_text)


    def load_and_display_dice_stats(self, csv_path):
        myocardium_values = []
        bloodpool_values = []

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                myocardium_values.append(float(row['Myocardium Dice']))
                bloodpool_values.append(float(row['Blood pool dice']))

        myocardium_mean = statistics.mean(myocardium_values)
        myocardium_std = statistics.stdev(myocardium_values)
        bloodpool_mean = statistics.mean(bloodpool_values)
        bloodpool_std = statistics.stdev(bloodpool_values)

        result_text = (
            f"Myocardium Dice:  Mean = {myocardium_mean:.3f},  Std Dev = {myocardium_std:.3f}\n"
            f"Blood Pool Dice:  Mean = {bloodpool_mean:.3f},  Std Dev = {bloodpool_std:.3f}"
        )

        self.dice_box.setPlainText(result_text)
