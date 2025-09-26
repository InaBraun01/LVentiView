from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, 
    QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QFont, QPalette, QLinearGradient, QBrush, QPainter, QPainterPath
from PyQt5.QtCore import Qt, QRect

class ModernButton(QPushButton):
    def __init__(self, text, icon_text="", primary=True):
        super().__init__()
        self.setText(text)
        self.icon_text = icon_text
        self.primary = primary
        self.setFixedSize(200, 120)
        self.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        
        # Modern styling
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4A90E2, stop:1 #357ABD);
                    border: none;
                    border-radius: 12px;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5BA0F2, stop:1 #458ACD);
                    transform: translateY(-2px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3A80D2, stop:1 #2F6AAD);
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f8f9fa, stop:1 #e9ecef);
                    border: 2px solid #dee2e6;
                    border-radius: 12px;
                    color: #495057;
                    padding: 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e9ecef, stop:1 #dee2e6);
                    border: 2px solid #4A90E2;
                    color: #4A90E2;
                }
                QPushButton:pressed {
                    background: #dee2e6;
                }
            """)
        
        # Add subtle shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)

class ModernCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 16px;
                border: none;  /* remove the border */
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(Qt.lightGray)
        self.setGraphicsEffect(shadow)

class StartPage(QWidget):
    def __init__(self, switch_to_analysis_callback, switch_to_mesh_callback):
        super().__init__()
        self.switch_to_analysis = switch_to_analysis_callback
        self.switch_to_mesh = switch_to_mesh_callback


        self.setAutoFillBackground(True)  # ensure the widget paints its background
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)
        
        # Set modern background
        self.setStyleSheet("""
            StartPage {
                background-color: transparent;
            }
            QWidget {
                background-color:transparent;
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
        """)


        # Main layout with better spacing
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(60, 50, 60, 50)
        main_layout.setSpacing(30)
        self.setLayout(main_layout)

        # Header section
        self.create_header(main_layout)
        
        # Action buttons section
        self.create_action_buttons(main_layout)
        
        # Information section
        self.create_info_section(main_layout)

    def create_header(self, main_layout):
        # Title with modern typography
        title = QLabel("LVentiView")
        title.setFont(QFont("Segoe UI", 32, QFont.Bold))  # Bold font
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #000000;  /* Black color */
                margin: 20px 0;
            }
        """)
        main_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Cardiac MRI Analysis & 3D Modeling Platform")
        subtitle.setFont(QFont("Segoe UI", 14, QFont.Normal))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                color: #000000;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(subtitle)

        # Description with better formatting
        description = QLabel(
            "Advanced tools for quantifying left ventricular volume (LVV), ejection fraction (EF), "
            "and regional myocardial thickness through automated segmentation and 3D mesh generation."
        )
        description.setFont(QFont("Segoe UI", 12))
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            QLabel {
                color: #5d6d7e;
                line-height: 1.4;
                max-width: 600px;
                margin: 0 auto 30px auto;
            }
        """)
        main_layout.addWidget(description)

    def create_action_buttons(self, main_layout):
        # Container for buttons with modern card design
        button_container = ModernCard()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(40, 30, 40, 30)
        button_layout.setSpacing(20)
        
        # Section title
        section_title = QLabel("Choose Your Module")
        section_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        section_title.setAlignment(Qt.AlignCenter)
        section_title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        button_layout.addWidget(section_title)

        # Buttons in horizontal layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(30)
        button_layout.addLayout(btn_layout)
        btn_layout.addStretch()

        # Segmentation button with icon
        btn_segmentation = ModernButton("Segmentation", primary=True)
        btn_segmentation.clicked.connect(self.on_segmentation)
        btn_layout.addWidget(btn_segmentation)

        # Mesh Generation button
        btn_mesh_gen = ModernButton("Mesh Generation", primary=True)
        btn_mesh_gen.clicked.connect(self.on_mesh_generation)
        btn_layout.addWidget(btn_mesh_gen)

        btn_layout.addStretch()
        
        main_layout.addWidget(button_container)

    def create_info_section(self, main_layout):
        # Information card
        info_card = ModernCard()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(30, 25, 30, 25)
        
        # Scrollable area for detailed instructions
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        scroll_area.setMaximumHeight(400)
        info_layout.addWidget(scroll_area)

        text_widget = QLabel(self.get_detailed_text())
        text_widget.setFont(QFont("Segoe UI", 11))
        text_widget.setWordWrap(True)
        text_widget.setAlignment(Qt.AlignTop)
        text_widget.setTextFormat(Qt.RichText)
        text_widget.setStyleSheet("""
            QLabel {
                background: transparent;
                line-height: 1.5;
                color: #2c3e50;
                padding: 15px;
            }
        """)
        scroll_area.setWidget(text_widget)
        
        main_layout.addWidget(info_card)
        main_layout.addStretch()

    def get_detailed_text(self):
        return """
        <div style="font-family: 'Segoe UI';">
        
        <h3 style="color: #4A90E2; font-size: 16px; margin-top: 0; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px;">
        Segmentation Module
        </h3>
        
        <p style="margin-bottom: 15px; line-height: 1.6;">
        The <strong>Segmentation Module</strong> provides automated processing of cardiac MRI series with minimal user interaction. Simply select your input folder and output directory, then initiate the analysis.
        </p>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A90E2;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50; margin-bottom: 8px;">Key Features:</p>
        <ul style="margin: 0; padding-left: 20px;">
        <li style="margin-bottom: 6px;"><strong>Data cleaning:</strong> Automatic removal of incomplete time points and slices outside the mitral valve–apex range</li>
        <li style="margin-bottom: 6px;"><strong>Volume analysis:</strong> Precise calculation of myocardial and blood pool volumes</li>
        <li style="margin-bottom: 6px;"><strong>Cardiac cycle analysis:</strong> Automated ED/ES state identification with volume curve visualization</li>
        </ul>
        </div>
        
        <h3 style="color: #4A90E2; font-size: 16px; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px;">
        Mesh Generation Module
        </h3>
        
        <p style="margin-bottom: 15px; line-height: 1.6;">
        The <strong>Mesh Generation Module</strong> creates detailed 3D left ventricular models from segmented MRI data. Load your segmentation data (.pkl files) and generate comprehensive 3D cardiac models.
        </p>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4A90E2;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50; margin-bottom: 8px;">Advanced Capabilities:</p>
        <ul style="margin: 0; padding-left: 20px;">
        <li style="margin-bottom: 6px;"><strong>3D modeling:</strong> High-fidelity mesh construction from segmentation data</li>
        <li style="margin-bottom: 6px;"><strong>Volume analysis:</strong> Comprehensive myocardial and blood pool volume computation</li>
        <li style="margin-bottom: 6px;"><strong>Thickness mapping:</strong> Detailed local myocardial thickness analysis across cardiac cycles</li>
        </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-top: 20px;">
        <p style="margin: 0; text-align: center; font-weight: 500;">
        Both modules feature fully configurable workflows with customizable parameters and comprehensive analysis options
        </p>
        </div>
        
        </div>
        """

    def on_segmentation(self):
        self.switch_to_analysis()

    def on_mesh_generation(self):
        self.switch_to_mesh()

    