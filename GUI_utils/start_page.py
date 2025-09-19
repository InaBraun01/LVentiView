import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from .icon_text_button import IconTextButton  # relative import

class StartPage(QWidget):
    def __init__(self, switch_to_analysis_callback, switch_to_mesh_callback):
        super().__init__()
        self.switch_to_analysis = switch_to_analysis_callback
        self.switch_to_mesh = switch_to_mesh_callback


        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addStretch()
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(50)
        main_layout.addLayout(btn_layout)
        btn_layout.addStretch()
        seg_icon_path = "segmentation_logo.png"
        mesh_icon_path = "mesh_gen_logo.png"
        btn_segmentation = IconTextButton(seg_icon_path, "Segmentation", size=180)
        btn_segmentation.clicked.connect(self.on_segmentation)
        btn_layout.addWidget(btn_segmentation)
        btn_mesh_gen = IconTextButton(mesh_icon_path, "Mesh Generation", size=180)
        btn_mesh_gen.clicked.connect(self.on_mesh_generation)
        btn_layout.addWidget(btn_mesh_gen)
        btn_layout.addStretch()
        main_layout.addStretch()

    def on_segmentation(self):
        self.switch_to_analysis()
    def on_mesh_generation(self):
        self.switch_to_mesh()


    