import sys
from PyQt5.QtWidgets import QApplication, QTextEdit, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QColor
import cv2
import numpy as np
import math

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
        self.operation_combo.addItem('Gray Conversion')
        self.operation_combo.addItem('Histogram Stretching')
        self.operation_combo.addItem('Histogram Widening')
        
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
        operation_texts = [
            r"İşlem Seç",
            r"""Gri Dönüşüm

Bu işlem, renkli bir görüntüyü gri tonlamalı bir görüntüye dönüştürür.

Her pikselin kırmızı, yeşil ve mavi kanallarının ağırlıklı toplamını hesaplayarak bunu yapar. Kullanılan formül:

    `0.2989 * R + 0.587 * G + 0.114 * B`.
""",
            r"""Histogram Germe

Bu işlem, görüntüdeki piksel yoğunluklarını belirli bir aralığa gererek kontrastı arttırmayı amaçlar.

İlk olarak, görüntüdeki minimum ve maksimum yoğunluk değerleri (c ve d) bulunur. Daha sonra, bu değerler arasındaki yoğunlukları, belirtilen aralığa (genellikle 0 ile 255 arasında) yerleştirmek için bir dönüşüm yapılır.

Formül şu şekildedir:

    P_{çıkış} = (P_{giriş} - c) * ((b - a) / (d - c)) + a

Burada:
- P_{giriş}: Giriş piksel değeri,
- P_{çıkış}: Çıkış piksel değeri,
- c ve d: Görüntüdeki minimum ve maksimum yoğunluk değerleri,
- a ve b: Çıkış aralığının minimum ve maksimum değerleri (genellikle 0 ve 255).

Sonuçta, kontrastı artırılmış ve daha geniş bir yoğunluk aralığına yayılmış bir görüntü elde edilir.
""",
            r"""Histogram Genişletme

gerekli bilgiler...
""",
        ]
        if index == 0:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Please select an operation.")
        else:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(blue_color)
            self.terminal_codes.append(operation_texts[index])
            
    def upload_image(self):
        file_name = "result\image2.jpg"
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
                
            elif operation == 'Gray Conversion':
                gray_image = self.convert_to_gray(self.image)
                self.terminal_codes.clear()
                self.download_image(gray_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Gray Conversion operation was applied.")
                
            elif operation == 'Histogram Stretching':
                stretched_image = self.histogram_stretching(self.image)
                self.terminal_codes.clear()
                self.download_image(stretched_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Histogram Stretching operation was applied.")

            elif operation == 'Histogram Widening':
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
        
        if self.operation_combo.currentText() == 'Gray Conversion':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Histogram Stretching':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Histogram Widening':
            # kodlar
            pass

        self.terminal_codes.clear()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Image downloaded as result.png.")
        
######################################## O P E R A S Y O N L A R ########################################
    
    def convert_to_gray(self, image):
        gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray_image[i, j] = int(0.2989 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0])
        return gray_image
    
    def histogram_stretching(self, image):
        if len(image.shape) == 3:
            gray = self.convert_to_gray(image)
        else:
            gray = image

        c = np.min(gray)
        d = np.max(gray)
        a, b = 0, 255

        stretched = (gray - c) * ((b - a) / (d - c)) + a
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)

        return stretched
    
    def histogram_widening(self, image):
        # kodlar
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())