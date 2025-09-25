import sys
from PyQt5.QtWidgets import QApplication
from GUI_utils.main_window import MainWindow

# This is the main for the GUI. 
# If you want to run the GUI from the terminal. 
# Go into the folder that this file is in. Activate the conda envrionment and run LVentiView.py

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()