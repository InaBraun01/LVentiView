import os
import glob
import csv
import statistics
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton,
    QScrollArea, QGroupBox, QTabWidget, QFormLayout, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QFrame, QGraphicsDropShadowEffect,
    QTextEdit, QPlainTextEdit, QSizePolicy, QListWidget, QFileDialog,
    QApplication, QProgressBar, QListWidgetItem
)
from PyQt5.QtGui import QFont, QIntValidator, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from .mesh_generation import (
    analyze_mesh_volumes_step,
    calculate_segmentation_thickness_step
)
from .segmentation_step_thread import AnalysisStepThread
from .mesh_fitting_thread import MeshGenerationThread


class ModernCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: none;
                margin: 5px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(Qt.lightGray)
        self.setGraphicsEffect(shadow)

class ModernButton(QPushButton):
    def __init__(self, text, primary=True, size="normal"):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        
        if size == "large":
            self.setFixedHeight(45)
            self.setFont(QFont("Segoe UI", 11, QFont.Bold))
        else:
            self.setFixedHeight(35)
            self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4A90E2, stop:1 #357ABD);
                    border: none;
                    border-radius: 8px;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5BA0F2, stop:1 #458ACD);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3A80D2, stop:1 #2F6AAD);
                }
                QPushButton:disabled {
                    background: #bdc3c7;
                    color: #7f8c8d;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    border: 2px solid #dee2e6;
                    border-radius: 8px;
                    color: #495057;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border-color: #4A90E2;
                    color: #4A90E2;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
            
            
class ZoomableImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)
# --- Main Application Class ---

class MeshGenerationApp(QWidget):
    backRequested = pyqtSignal()

    def __init__(self, folder_manager):
        super().__init__()
        
        self.folder_manager = folder_manager
        self.resize(900, 900)
        
        # Initialize all instance variables
        self.input_path = None
        self.output_folder = None
        self.param_fields = {}
        self.pending_analysis_steps = []
        self.current_analysis_step = None
        self.mesh_thread = None
        self.analysis_step_thread = None
        self.buttons_created = False
        self.volumetric_mesh_checked = True
        self.thickness_checked = False

        self.setStyleSheet("""
            MeshGenerationApp { background-color: #f8f9fa; }
            QWidget { background-color: transparent; }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical { border: none; background: #f1f3f4; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #c1c1c1; border-radius: 4px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #a1a1a1; }
            QGroupBox { font: 14px "Segoe UI"; font-weight: 600; color: #2c3e50; border: 2px solid #dee2e6; border-radius: 8px; margin-top: 10px; padding-top: 10px; background-color: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 8px 0 8px; background-color: white; }
            QTabWidget::pane { border: 1px solid #dee2e6; border-radius: 6px; background-color: white; }
            QTabBar::tab { background: #f8f9fa; border: 1px solid #dee2e6; border-bottom: none; border-radius: 6px 6px 0 0; padding: 8px 16px; margin-right: 2px; color: #495057; font: 12px "Segoe UI"; }
            QTabBar::tab:selected { background: white; color: #4A90E2; font-weight: 600; }
            QTabBar::tab:hover { background: #e9ecef; }
            QCheckBox { font: 12px "Segoe UI"; font-weight: 500; color: #2c3e50; spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #dee2e6; border-radius: 3px; background-color: white; }
            QCheckBox::indicator:checked { background-color: #4A90E2; border-color: #4A90E2; }
            QLineEdit, QComboBox { border: 2px solid #dee2e6; border-radius: 6px; padding: 8px 12px; font: 14px "Segoe UI"; background-color: white; }
            QLineEdit:focus, QComboBox:focus { border-color: #4A90E2; outline: none; }
            QComboBox::drop-down { border: none; }
            QTextEdit, QPlainTextEdit { border: 2px solid #dee2e6; border-radius: 8px; padding: 12px; font: 14px "Segoe UI"; background-color: white; color: #2c3e50; }
            QListWidget { border: 2px solid #dee2e6; border-radius: 8px; background-color: white; alternate-background-color: #f8f9fa; font: 12px "Segoe UI"; padding: 4px; }
            QListWidget::item { border-radius: 4px; padding: 6px 8px; margin: 1px; }
            QListWidget::item:selected { background-color: #4A90E2; color: white; }
            QListWidget::item:hover { background-color: #e3f2fd; }
        """)

        # Create main layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(20)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        self.layout = QVBoxLayout(scroll_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(20)

        # Create all UI sections
        self.create_header()
        self.setup_folder_ui()
        self.create_parameters_section()
        self.create_analysis_controls()
        self.create_run_controls()
        self.create_results_section()

        # Connect signals
        self.folder_manager.inputFolderChanged.connect(self.update_input_label)
        self.folder_manager.outputFolderChanged.connect(self.update_output_label)
        
        # Initialize labels
        self.update_input_label(self.folder_manager.get_input_folder())
        self.update_output_label(self.folder_manager.get_output_folder())
        
        # Connect button and widget signals
        self.run_button.clicked.connect(self.run_mesh_generation)
        self.volumetric_mesh_checkbox.stateChanged.connect(self.on_volumetric_mesh_checked)
        self.thickness_checkbox.stateChanged.connect(self.on_thickness_checked)
        self.mesh_image_list.itemChanged.connect(self.on_image_checked)
        self.volumetric_analysis_list.itemChanged.connect(self.on_image_checked)
        self.thickness_analysis_list.itemChanged.connect(self.on_image_checked)
        
        # Set initial state
        self.update_analysis_widgets_visibility()
        self.check_enable_run()

    def create_header(self):
        header_card = ModernCard()
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(25, 20, 25, 20)
        
        top_layout = QHBoxLayout()
        self.back_button = QToolButton()
        self.back_button.setArrowType(Qt.LeftArrow)
        self.back_button.setToolTip("Back to Start Page")
        self.back_button.setFixedSize(35, 35)
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.clicked.connect(self.backRequested.emit)
        self.back_button.setStyleSheet("""
            QToolButton { border: 2px solid #dee2e6; border-radius: 8px; background-color: white; color: #000000; }
            QToolButton:hover { border-color: #4A90E2; color: #000000; background-color: #f8f9fa; }
        """)
        
        title = QLabel("Mesh Generation Module")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #000000; margin-left: 15px;")
        
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(title)
        top_layout.addStretch()
        
        description = QLabel("Fitting of a 3D mesh to segmentation masks, mesh based volume and myocardial thickness calculation")
        description.setFont(QFont("Segoe UI", 12))
        description.setStyleSheet("color: #000000; margin-top: 8px;")
        description.setWordWrap(True)
        
        header_layout.addLayout(top_layout)
        header_layout.addWidget(description)
        self.layout.addWidget(header_card)

    def setup_folder_ui(self):
        folder_card = ModernCard()
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(25, 20, 25, 20)
        folder_layout.setSpacing(15)
        
        section_title = QLabel("Data Selection")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #000000; margin-bottom: 5px;")
        folder_layout.addWidget(section_title)
        
        # Input folder selection
        input_layout = QHBoxLayout()
        input_layout.setSpacing(15)
        self.input_label = QLabel("No segmented data folder selected")
        self.input_label.setStyleSheet("""
            QLabel { 
                color: #495057; 
                font: 14px "Segoe UI"; 
                padding: 8px; 
                background-color: #f8f9fa; 
                border: 2px solid #dee2e6; 
                border-radius: 6px; 
            }
        """)
        btn_pick_input = ModernButton("Select Segmented Data...", primary=False)
        btn_pick_input.clicked.connect(self.pick_input_folder)
        input_layout.addWidget(self.input_label, stretch=1)
        input_layout.addWidget(btn_pick_input)
        folder_layout.addLayout(input_layout)
        
        # Output folder selection
        output_layout = QHBoxLayout()
        output_layout.setSpacing(15)
        self.output_label = QLabel("No mesh output folder selected")
        self.output_label.setStyleSheet("""
            QLabel { 
                color: #495057; 
                font: 14px "Segoe UI"; 
                padding: 8px; 
                background-color: #f8f9fa; 
                border: 2px solid #dee2e6; 
                border-radius: 6px; 
            }
        """)
        btn_pick_output = ModernButton("Browse Mesh Output...", primary=False)
        btn_pick_output.clicked.connect(self.pick_output_folder)
        output_layout.addWidget(self.output_label, stretch=1)
        output_layout.addWidget(btn_pick_output)
        folder_layout.addLayout(output_layout)
        
        self.layout.addWidget(folder_card)

    def create_parameters_section(self):
        self.set_params_group = QGroupBox()
        self.layout.addWidget(self.set_params_group)
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(15, 20, 15, 15)
        self.set_params_group.setLayout(group_layout)
        
        self.params_checkbox = QCheckBox("Set Advanced Mesh Parameters")
        self.params_checkbox.setChecked(False)
        self.params_checkbox.setStyleSheet("QCheckBox { font-size: 14px; color: #2c3e50; }")
        self.params_checkbox.stateChanged.connect(self.toggle_params_visibility)
        group_layout.addWidget(self.params_checkbox)
        
        self.set_params_tab_widget = QTabWidget()
        group_layout.addWidget(self.set_params_tab_widget)
        self.set_params_tab_widget.setVisible(False)
        
        # Mesh fitting parameters tab
        fitting_params_widget = QWidget()
        fitting_params_layout = QFormLayout(fitting_params_widget)
        fitting_params_layout.setSpacing(12)
        
        fitting_params = [
            ('time_frames_to_fit', 'Time Frames to be Fit', 'all'),
            ('fitting_steps', 'fitting Steps', '1'),
            ('lr', 'Learning Rate', '0.003'),
            ('num_modes', 'Number of Modes', '25'),
        ]
        
        for key, label, default in fitting_params:
            if key == 'time_frames_to_fit':
                input_widget = QComboBox()
                input_widget.setEditable(True)
                input_widget.addItems(['all'])
                input_widget.setEditText(default)
                input_widget.lineEdit().setPlaceholderText("Enter 'all' or comma-separated positive integers (e.g., 1,2,3)")
            else:
                input_widget = QLineEdit()
                input_widget.setText(default)
                if key in ['num_modes', 'fitting_steps']:
                    input_widget.setValidator(QIntValidator(0, 1000000, self))
            
            label_widget = QLabel(label + ':')
            label_widget.setStyleSheet("font-weight: 500; color: #2c3e50;")
            fitting_params_layout.addRow(label_widget, input_widget)
            self.param_fields[key] = input_widget
        
        self.set_params_tab_widget.addTab(fitting_params_widget, "Mesh Fitting Parameters")
        
        # Advanced fitting parameters tab
        advanced_fitting_params_widget = QWidget()
        advanced_fitting_layout = QFormLayout(advanced_fitting_params_widget)
        
        advanced_fitting_params = [
            ('cp_frequency', 'Control Point Frequency', '50'),
            ('mesh_model_dir', 'Shape Model Directory', 'ShapeModel'),
            ('steps_between_fig_saves', 'Steps between Mesh Update', '50')
        ]
        
        for key, label, default in advanced_fitting_params:
            input_widget = QLineEdit()
            input_widget.setText(default)
            if key in ['cp_frequency', 'steps_between_fig_saves']:
                input_widget.setValidator(QIntValidator(0, 1000000, self))
            advanced_fitting_layout.addRow(label + ':', input_widget)
            self.param_fields[key] = input_widget
        
        self.set_params_tab_widget.addTab(advanced_fitting_params_widget, "Advanced Fitting Parameters")

        # Shift and Rotation parameters tab
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

        # Weight parameters tab
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

    def create_analysis_controls(self):
        controls_card = ModernCard()
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(25, 20, 25, 20)
        controls_layout.setSpacing(15)

        section_title = QLabel("Select Post-Mesh Analysis")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #000000; margin-bottom: 5px;")
        controls_layout.addWidget(section_title)

        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(10)

        self.volumetric_mesh_checkbox = QCheckBox("Calculate Volumes")
        self.volumetric_mesh_checkbox.setChecked(True)
        self.volumetric_mesh_checkbox.setStyleSheet("QCheckBox { font-size: 14px; color: #2c3e50; }")
        checkbox_layout.addWidget(self.volumetric_mesh_checkbox)

        self.thickness_checkbox = QCheckBox("Calculate Thickness")
        self.thickness_checkbox.setChecked(False)
        self.thickness_checkbox.setStyleSheet("QCheckBox { font-size: 14px; color: #2c3e50; }")
        checkbox_layout.addWidget(self.thickness_checkbox)

        controls_layout.addLayout(checkbox_layout)
        self.layout.addWidget(controls_card)

    def create_run_controls(self):
        controls_card = ModernCard()
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(25, 20, 25, 20)
        controls_layout.setSpacing(15)
        
        self.progressBar = QProgressBar()
        self.progressBar.setTextVisible(True)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                text-align: center;
                color: #2c3e50;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 6px;
            }
        """)
        controls_layout.addWidget(self.progressBar)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.run_button = ModernButton("Run Mesh Generation", primary=True, size="large")
        self.run_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)
        
        self.layout.addWidget(controls_card)

    def create_results_section(self):
        results_card = ModernCard()
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(25, 20, 25, 20)
        results_layout.setSpacing(15)

        section_title = QLabel("Mesh Generation Progress & Results")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        results_layout.addWidget(section_title)

        # Log output
        log_label = QLabel("Mesh Log:")
        log_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        log_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        results_layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)
        self.log_output.setPlaceholderText("Mesh generation log will appear here...")
        self.log_output.setStyleSheet("""
            QTextEdit {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
                color: #2c3e50;
                font: 14px "Segoe UI";
            }
        """)
        results_layout.addWidget(self.log_output)

        # Content layout with lists and image display
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Lists layout
        lists_layout = QVBoxLayout()
        lists_layout.setSpacing(15)

        # Mesh visualization list
        self.label_mesh_image = QLabel("Mesh Visualization:")
        self.label_mesh_image.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.label_mesh_image.setStyleSheet("color: #495057; margin-bottom: 5px;")
        lists_layout.addWidget(self.label_mesh_image)
        
        self.mesh_image_list = QListWidget()
        self.mesh_image_list.setMaximumHeight(100)
        lists_layout.addWidget(self.mesh_image_list)

        # Volumetric analysis list
        self.volumetric_analysis_label = QLabel("Blood Pool and Myocardium Volumes:")
        self.volumetric_analysis_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.volumetric_analysis_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        lists_layout.addWidget(self.volumetric_analysis_label)
        
        self.volumetric_analysis_list = QListWidget()
        self.volumetric_analysis_list.setMaximumHeight(100)
        lists_layout.addWidget(self.volumetric_analysis_list)

        # Thickness analysis list
        self.thickness_analysis_label = QLabel("Local Thickness Estimation:")
        self.thickness_analysis_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.thickness_analysis_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        lists_layout.addWidget(self.thickness_analysis_label)
        
        self.thickness_analysis_list = QListWidget()
        self.thickness_analysis_list.setMaximumHeight(100)
        lists_layout.addWidget(self.thickness_analysis_list)

        content_layout.addLayout(lists_layout, stretch=1)

        # Image display
        display_layout = QVBoxLayout()
        display_label = QLabel("Figure Preview:")
        display_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        display_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        display_layout.addWidget(display_label)
        
        self.image_display = ZoomableImageLabel()
        self.image_display.setText("No image selected")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display.setMinimumHeight(250)
        self.image_display.setStyleSheet("""
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            color: #7f8c8d;
            font: 11px "Segoe UI";
        """)

        display_layout.addWidget(self.image_display)

        content_layout.addLayout(display_layout, stretch=2)
        results_layout.addLayout(content_layout)

        # Results text boxes
        metrics_layout = QVBoxLayout()
        
        # Dice scores
        self.dice_title = QLabel("Fit of Meshes to Segmentation Masks:")
        self.dice_title.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.dice_title.setStyleSheet("color: #495057; margin-bottom: 5px; margin-top: 15px;")
        metrics_layout.addWidget(self.dice_title)
        
        self.dice_box = QPlainTextEdit()
        self.dice_box.setReadOnly(True)
        self.dice_box.setMaximumHeight(80)
        self.dice_box.setPlaceholderText("Dice score results will appear here...")
        self.dice_box.setStyleSheet("""
            QPlainTextEdit {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
                color: #2c3e50;
                font: 14px "Segoe UI";
            }
        """)
        metrics_layout.addWidget(self.dice_box)
        
        # Computed cardiac function parameters
        self.results_title = QLabel("Computed Cardiac Function Parameters:")
        self.results_title.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.results_title.setStyleSheet("color: #495057; margin-bottom: 5px; margin-top: 15px;")
        metrics_layout.addWidget(self.results_title)
        
        self.results_box = QPlainTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setMaximumHeight(120)
        self.results_box.setPlaceholderText("Volume and thickness calculations will appear here after analysis...")
        self.results_box.setStyleSheet("""
            QPlainTextEdit {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
                color: #2c3e50;
                font: 14px "Segoe UI";
            }
        """)
        metrics_layout.addWidget(self.results_box)
        
        
        results_layout.addLayout(metrics_layout)
        self.layout.addWidget(results_card)

    def get_file_dialog_style(self):
        return """
            QFileDialog {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: white;
                color: black;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f3f4;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a1a1a1;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: black;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """

    def toggle_params_visibility(self, checked):
        self.set_params_tab_widget.setVisible(checked)

        # Folder picker methods
    def pick_input_folder(self):
        dialog = QFileDialog(self, "Select MRI Data Folder")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet(self.get_file_dialog_style())

        if dialog.exec_():
            folder = dialog.selectedFiles()[0]
            self.folder_manager.set_input_folder(folder)
    
    def pick_output_folder(self):
        dialog = QFileDialog(self, "Select Output Folder")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet(self.get_file_dialog_style())

        if dialog.exec_():
            folder = dialog.selectedFiles()[0]
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
            'num_cycle', 'num_modes', 'fitting_steps', 'steps_between_progress_update',
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
                    elif key in ['fitting_steps']:
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
        try:
            patterns = ("*.pdf","*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
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

        except Exception as e:
            self.log_output.append(f"Error processing analysis plots: {str(e)}")

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