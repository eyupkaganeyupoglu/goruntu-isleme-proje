import sys
from PyQt5.QtWidgets import QApplication, QTextEdit, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QColor
import cv2
import numpy as np
import math
import all_operation_texts

red_color = QColor(204, 0, 0)
green_color = QColor(0, 153, 0)
blue_color = QColor(0, 102, 255)
black_color = QColor(0, 0, 0)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None

    def initUI(self):
        self.layout = QVBoxLayout()

        self.terminal_codes = QTextEdit(self)
        self.terminal_codes.setReadOnly(True)
        self.layout.addWidget(self.terminal_codes)
        self.terminal_codes.setFixedSize(800,300)
        
        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem('Select Operation')
        self.operation_combo.addItem('Edge Detection Prewitt')
        
        self.layout.addWidget(self.operation_combo)

        self.apply_button = QPushButton('Apply and Download', self)
        self.apply_button.clicked.connect(self.apply_operation)
        self.layout.addWidget(self.apply_button)

        self.setLayout(self.layout)

        self.show()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Program is ready.")
        
        self.operation_combo.currentIndexChanged.connect(self.update_terminal_codes)
        
########################################  T E M E L    F O N K S İ Y O N L A R  ########################################

    def update_terminal_codes(self, index):
        operation_texts = all_operation_texts.operation_texts
        
        if index == 0:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Please select an operation.")
        else:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(blue_color)
            self.terminal_codes.append(operation_texts[index])
            
    def upload_image(self):
        file_name = "result\image.jpg"
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(green_color)
            self.terminal_codes.append("Source image uploaded.")
    
    def apply_operation(self):
        
        if self.image is None:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Please upload an image first before applying any operation.")
            return
        
        operation = self.operation_combo.currentText()
        
        try:
            if operation == 'Select Operation':
                self.terminal_codes.clear()
                self.terminal_codes.setTextColor(red_color)
                self.terminal_codes.append("Please select an operation.")

            elif operation == 'Edge Detection Prewitt':
                # kodlar
                pass

        except Exception as e:
            self.terminal_codes.append("Error occurred while applying '{}' operation:\n\n{}".format(operation, str(e)))
        finally:
            pass
        
    def download_image(self, image):
        if self.image is None:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Please apply an operation first before downloading the image.")
            return
        
        elif self.operation_combo.currentText() == 'Edge Detection Prewitt':
            # kodlar
            pass

        self.terminal_codes.clear()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Image downloaded as result.png.")
        
######################################## O P E R A S Y O N L A R ########################################
    
    def edge_detection_prewitt(self):
        # kodlar
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())