from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

class StartPage(QWidget):
    def __init__(self, switch_to_analysis_callback, switch_to_mesh_callback):
        super().__init__()
        self.switch_to_analysis = switch_to_analysis_callback
        self.switch_to_mesh = switch_to_mesh_callback

        # Main vertical layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addStretch()

        # Horizontal layout for buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(50)
        main_layout.addLayout(btn_layout)
        btn_layout.addStretch()

        # Segmentation button
        btn_segmentation = QPushButton("Segmentation")
        btn_segmentation.setFixedSize(180, 180)  # square closed button
        btn_segmentation.clicked.connect(self.on_segmentation)
        btn_layout.addWidget(btn_segmentation)

        # Mesh Generation button
        btn_mesh_gen = QPushButton("Mesh Generation")
        btn_mesh_gen.setFixedSize(180, 180)  # square closed button
        btn_mesh_gen.clicked.connect(self.on_mesh_generation)
        btn_layout.addWidget(btn_mesh_gen)

        btn_layout.addStretch()
        main_layout.addStretch()

    def on_segmentation(self):
        self.switch_to_analysis()

    def on_mesh_generation(self):
        self.switch_to_mesh()


    