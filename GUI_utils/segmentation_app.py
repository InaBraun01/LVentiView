import os
import csv
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem, 
                             QSizePolicy, QCheckBox, QToolButton, QGroupBox, QTabWidget,
                             QFormLayout, QLineEdit, QScrollArea, QPlainTextEdit, QFrame,
                             QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QIntValidator, QValidator, QFont, QPalette
from .segmentation_thread import AnalysisThread
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

class IntListValidator(QValidator):
    def validate(self, input_str, pos):
        # Allow empty input
        if input_str.strip() == "":
            return QValidator.Intermediate, input_str, pos

        # Split on commas
        parts = input_str.split(",")
        for part in parts:
            part = part.strip()
            if part == "":
                # Allow trailing comma while typing (e.g. "1,2,")
                continue
            if not part.isdigit() or int(part) <= 0:
                return QValidator.Invalid, input_str, pos

        return QValidator.Acceptable, input_str, pos

    def fixup(self, input_str):
        # Auto-clean invalid input: keep only positive numbers
        parts = [p.strip() for p in input_str.split(",") if p.strip().isdigit() and int(p.strip()) > 0]
        return ", ".join(parts)

class ModernCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: none;  /* remove the border */
                margin: 5px;
            }
        """)
        
        # Add shadow effect
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

class DicomAnalysisApp(QWidget):
    backRequested = pyqtSignal()
    switchToMeshRequested = pyqtSignal()

    def __init__(self, folder_manager):
        super().__init__()
        self.folder_manager = folder_manager
        self.resize(1000, 900)
        
        # Set modern styling
        self.setStyleSheet("""
            DicomAnalysisApp {
                background-color: #f8f9fa;
            }
            QWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
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
            QGroupBox {
                font: 14px "Segoe UI";
                font-weight: 600;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                background-color: white;
            }
            QTabBar::tab {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
                color: #495057;
                font: 12px "Segoe UI";
            }
            QTabBar::tab:selected {
                background: white;
                color: #4A90E2;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background: #e9ecef;
            }
            QCheckBox {
                font: 12px "Segoe UI";
                font-weight: 500;
                color: #2c3e50;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #dee2e6;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #4A90E2;
                border-color: #4A90E2;
            }
            QLineEdit {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 12px;
                font: 14px "Segoe UI";
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4A90E2;
                outline: none;
            }
            QTextEdit, QPlainTextEdit {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 12px;
                font: 14px "Segoe UI";
                background-color: white;
                color: #2c3e50;
            }
            QListWidget {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                alternate-background-color: #f8f9fa;
                font: 12px "Segoe UI";
                padding: 4px;
            }
            QListWidget::item {
                border-radius: 4px;
                padding: 6px 8px;
                margin: 1px;
            }
            QListWidget::item:selected {
                background-color: #4A90E2;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
        """)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        self.setLayout(main_layout)

        # --- Header with back button ---
        self.create_header(main_layout)
        
        # --- Main content in scroll area ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        self.layout = scroll_layout

        # --- Folder selection in card ---
        self.create_folder_selection()

        # --- Parameters section ---
        self.create_parameters_section()

        # --- Analysis controls ---
        self.create_analysis_controls()

        # --- Results section ---
        self.create_results_section()

        self.update_input_label(self.folder_manager.get_input_folder())
        self.update_output_label(self.folder_manager.get_output_folder())

        # --- Connect folder manager signals ---
        self.folder_manager.inputFolderChanged.connect(self.update_input_label)
        self.folder_manager.outputFolderChanged.connect(self.update_output_label)


        # --- State variables ---
        self.selected_input_folder = None
        self.selected_output_folder = None
        self.analysis_thread = None
        self.seg_image_folder = None
        self.cardiac_plot_folder = None

    def create_header(self, main_layout):
        header_card = ModernCard()
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(25, 20, 25, 20)
        
        # Top row with back button and title
        top_layout = QHBoxLayout()
        
        self.back_button = QToolButton()
        self.back_button.setArrowType(Qt.LeftArrow)
        self.back_button.setToolTip("Back to Start Page")
        self.back_button.setFixedSize(35, 35)
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.clicked.connect(self.backRequested.emit)
        self.back_button.setStyleSheet("""
            QToolButton {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                color: #000000;
            }
            QToolButton:hover {
                border-color: #4A90E2;
                color: #000000;
                background-color: #f8f9fa;
            }
        """)
        
        title = QLabel("Segmentation Module")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #000000; margin-left: 15px;")
        
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(title)
        top_layout.addStretch()
        
        # Description
        description = QLabel("Automated cardiac MRI segmentation, data cleaning and volume calculation using Simpson's Method")
        description.setFont(QFont("Segoe UI", 12))
        description.setStyleSheet("color: #000000; margin-top: 8px;")
        description.setWordWrap(True)
        
        header_layout.addLayout(top_layout)
        header_layout.addWidget(description)
        
        main_layout.addWidget(header_card)

    def create_folder_selection(self):
        folder_card = ModernCard()
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(25, 20, 25, 20)
        folder_layout.setSpacing(15)
        
        # Section title
        section_title = QLabel("Data Selection")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #000000; margin-bottom: 5px;")
        folder_layout.addWidget(section_title)
        
        # Input folder
        input_layout = QHBoxLayout()
        input_layout.setSpacing(15)
        self.input_label = QLabel("No MRI data folder selected")
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
        self.input_label.setAutoFillBackground(False)
        btn_pick_input = ModernButton("Browse MRI Data...", primary=False)
        btn_pick_input.clicked.connect(self.pick_input_folder)
        
        input_layout.addWidget(self.input_label, stretch=1)
        input_layout.addWidget(btn_pick_input)
        folder_layout.addLayout(input_layout)
        
        # Output folder
        output_layout = QHBoxLayout()
        output_layout.setSpacing(15)
        self.output_label = QLabel("No output folder selected")
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
        btn_pick_output = ModernButton("Browse Output...", primary=False)
        btn_pick_output.clicked.connect(self.pick_output_folder)
        
        output_layout.addWidget(self.output_label, stretch=1)
        output_layout.addWidget(btn_pick_output)
        folder_layout.addLayout(output_layout)
        
        self.layout.addWidget(folder_card)


    def create_parameters_section(self):
        # Dictionaries to store parameter input widgets
        self.param_fields = {}
        self.crop_param_fields = {}
        # Create a QGroupBox without a title
        self.set_params_group = QGroupBox()  # no title
        self.layout.addWidget(self.set_params_group)

        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(15, 20, 15, 15)
        self.set_params_group.setLayout(group_layout)

        # Add a large checkbox inside the group box
        self.params_checkbox = QCheckBox("Select Advanced Parameters")
        self.params_checkbox.setChecked(False)
        self.params_checkbox.stateChanged.connect(self.toggle_params_visibility)
        self.params_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;  /* make text larger */
                color: #2c3e50;   /* dark color */
            }
        """)
        group_layout.addWidget(self.params_checkbox)

        
        self.set_params_tab_widget = QTabWidget()
        group_layout.addWidget(self.set_params_tab_widget)
        self.set_params_tab_widget.setVisible(False)

        # Clean data parameters tab
        clean_params_widget = QWidget()
        clean_params_layout = QFormLayout(clean_params_widget)
        clean_params_layout.setSpacing(12)

        clean_params = [
            ('percentage_base', 'Exclusion threshold at base', '0.3'),
            ('percentage_apex', 'Exclusion threshold at apex', '0.2'),
            ('slice_threshold', 'Exclusion threshold of missing slices', '1'),
            ('margin_factor','Multiplicative padding factor', '2.5')
        ]

        for key, label, default in clean_params:
            input_widget = QLineEdit()
            input_widget.setText(default)
            
            # Set validators
            if key in ['percentage_base', 'percentage_apex']:
                input_widget.setValidator(PercentageValidator())
            elif key == 'slice_threshold':
                input_widget.setValidator(QIntValidator(0, 1000000, self))

            if key == 'margin_factor':
                self.crop_param_fields[key] = input_widget
            else:
                self.param_fields[key] = input_widget

            label_widget = QLabel(label + ':')
            label_widget.setStyleSheet("font-weight: 500; color: #2c3e50;")
            clean_params_layout.addRow(label_widget, input_widget)

        self.set_params_tab_widget.addTab(clean_params_widget, "MRI Cleaning")

        # Manual cleaning parameters tab
        manual_params_widget = QWidget()
        manual_params_layout = QFormLayout(manual_params_widget)
        manual_params_layout.setSpacing(12)
        
        manual_params = [
            ('remove_z_slices', 'Z Slices to Remove'),
            ('remove_time_steps', 'Time Steps to Remove')
        ]

        for key, label in manual_params:
            input_widget = QLineEdit()
            input_widget.setText("")
            input_widget.setValidator(IntListValidator())
            input_widget.setPlaceholderText("e.g., 1,2,5 or leave empty")
            
            label_widget = QLabel(label + ':')
            label_widget.setStyleSheet("font-weight: 500; color: #2c3e50;")
            manual_params_layout.addRow(label_widget, input_widget)
            self.param_fields[key] = input_widget

        self.set_params_tab_widget.addTab(manual_params_widget, "Manual Cleaning")

    def create_analysis_controls(self):
        controls_card = ModernCard()
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(25, 20, 25, 20)
        controls_layout.setSpacing(15)
        
        section_title = QLabel("Select Postprocessing Steps")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #000000; margin-bottom: 5px;")
        controls_layout.addWidget(section_title)

        # Checkboxes
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(10)
        
        self.clean_data_checkbox = QCheckBox("Enable Data Cleaning")
        self.clean_data_checkbox.setChecked(True)
        self.clean_data_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;  /* make text larger */
                color: #2c3e50;   /* dark color */
            }
        """)
        checkbox_layout.addWidget(self.clean_data_checkbox)

        self.cardiac_plots_checkbox = QCheckBox("Calculate Volume Metrics")
        self.cardiac_plots_checkbox.setChecked(True)
        self.cardiac_plots_checkbox.stateChanged.connect(self.toggle_cardiac_plots_visibility)
        self.cardiac_plots_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;  /* make text larger */
                color: #2c3e50;   /* dark color */
            }
        """)
        checkbox_layout.addWidget(self.cardiac_plots_checkbox)
        
        controls_layout.addLayout(checkbox_layout)

        # Run button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.btn_run = ModernButton("Run Segmentation Analysis", primary=True, size="large")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.btn_run)
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)
        
        self.layout.addWidget(controls_card)

    def create_results_section(self):
        results_card = ModernCard()
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(25, 20, 25, 20)
        results_layout.setSpacing(15)
        
        section_title = QLabel("Analysis Progress & Results")
        section_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        section_title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        results_layout.addWidget(section_title)

        # Log output
        log_label = QLabel("Analysis Log:")
        log_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        log_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        results_layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)
        self.log_output.setPlaceholderText("Analysis log will appear here...")
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

        # Two-column layout for lists and results
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left column - Lists
        lists_layout = QVBoxLayout()
        
        # Segmentation images
        self.label_segmentation = QLabel("Segmentation Results:")
        self.label_segmentation.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.label_segmentation.setStyleSheet("color: #495057; margin-bottom: 5px;")
        lists_layout.addWidget(self.label_segmentation)
        
        self.seg_image_list = QListWidget()
        self.seg_image_list.itemChanged.connect(self.on_seg_image_checked)
        self.seg_image_list.setMaximumHeight(100)
        lists_layout.addWidget(self.seg_image_list)

        # Cardiac plots
        self.label_cardiac = QLabel("Volume Analysis:")
        self.label_cardiac.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.label_cardiac.setStyleSheet("color: #495057; margin-bottom: 5px; margin-top: 10px;")
        lists_layout.addWidget(self.label_cardiac)
        
        self.cardiac_plot_list = QListWidget()
        self.cardiac_plot_list.itemChanged.connect(self.on_cardiac_plot_checked)
        self.cardiac_plot_list.setMaximumHeight(100)
        lists_layout.addWidget(self.cardiac_plot_list)

        # Hide cardiac plots initially
        self.label_cardiac.setVisible(self.cardiac_plots_checkbox.isChecked())
        self.cardiac_plot_list.setVisible(self.cardiac_plots_checkbox.isChecked())

        content_layout.addLayout(lists_layout, stretch=1)
        
        # Right column - Image display
        display_layout = QVBoxLayout()
        
        display_label = QLabel("Image Preview:")
        display_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        display_label.setStyleSheet("color: #495057; margin-bottom: 5px;")
        display_layout.addWidget(display_label)
        
        self.image_display = ZoomableImageLabel()
        self.image_display.setText("Select an image from the lists on the left")
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

        # Cardiac function results
        self.results_title = QLabel("Computed Cardiac Parameters:")
        self.results_title.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.results_title.setStyleSheet("color: #495057; margin-bottom: 5px; margin-top: 15px;")
        results_layout.addWidget(self.results_title)

        self.results_box = QPlainTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setMaximumHeight(120)
        self.results_box.setPlaceholderText("Cardiac parameters will appear here after analysis...")
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
        results_layout.addWidget(self.results_box)

        # Switch to mesh button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.to_mesh_button = ModernButton("Continue to Mesh Generation", primary=True, size="large")
        self.to_mesh_button.clicked.connect(self.switchToMeshRequested.emit)
        self.to_mesh_button.hide()
        button_layout.addWidget(self.to_mesh_button)
        button_layout.addStretch()
        results_layout.addLayout(button_layout)
        
        self.layout.addWidget(results_card)




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
        dialog = QFileDialog(self, "Select MRI Data Folder")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)  # Important: allow directory selection
        dialog.setOption(QFileDialog.ShowDirsOnly, True)  # Only show folders

        # Apply your light theme
        dialog.setStyleSheet("""
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
        """)

        if dialog.exec_():
            folder = dialog.selectedFiles()[0]  # Get the selected folder
            self.folder_manager.set_input_folder(folder)
            self.update_input_label(folder)
    
    def pick_output_folder(self):
        dialog = QFileDialog(self, "Select Output Folder")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)  # Important: allow directory selection
        dialog.setOption(QFileDialog.ShowDirsOnly, True)  # Only show folders

        # Apply your light theme
        dialog.setStyleSheet("""
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
        """)

        if dialog.exec_():
            folder = dialog.selectedFiles()[0]  # Get the selected folder
            self.folder_manager.set_output_folder(folder)
            self.update_output_label(folder)
    
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
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
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
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
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
            print(key)
            text = widget.text().strip()

            if key in ['percentage_base', 'percentage_apex']:
                # Floats in [0, 1]
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
                # Single integer
                try:
                    params[key] = int(text)
                except ValueError:
                    self.log_output.append(f"Invalid integer for {key}: {text}")
                    params[key] = 0  # Default fallback

            elif key.strip().lower() in ['remove_z_slices', 'remove_time_steps']:   
                # List of integers
                if text == "":
                    params[key] = []  # Empty allowed
                else:
                    try:
                        int_list = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
                        params[key] = int_list
                    except ValueError:
                        self.log_output.append(f"Invalid list of integers for {key}: {text}")
                        params[key] = []  # Default fallback

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
                value_str = row['Value']
                time_step = row['Time_step']

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
