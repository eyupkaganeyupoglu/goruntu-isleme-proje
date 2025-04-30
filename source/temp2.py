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
        self.terminal_codes.setFixedSize(400,150)
        
        self.upload_button = QPushButton('Resim Yükle', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem('İşlem Seçin')
        self.operation_combo.addItem('Gri Dönüşüm')
        self.operation_combo.addItem('Binary Dönüşüm')
        self.operation_combo.addItem('Görüntü Döndürme')
        
        self.layout.addWidget(self.operation_combo)

        self.apply_button = QPushButton('Uygula ve İndir', self)
        self.apply_button.clicked.connect(self.apply_operation)
        self.layout.addWidget(self.apply_button)

        self.setLayout(self.layout)

        self.show()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Program hazır.")
        
        self.operation_combo.currentIndexChanged.connect(self.update_terminal_codes)
        
########################################  T E M E L    F O N K S İ Y O N L A R  ########################################

    def update_terminal_codes(self, index):
        operation_texts = [
            "İşlem Seçin",
            """Gri Dönüşüm

Bu işlem, renkli bir resmi gri tonlamalı bir resme dönüştürür.

Bunu, her pikselin kırmızı, yeşil ve mavi kanallarının ağırlıklı toplamını hesaplayarak yapar. Formül şu şekildedir:
`0.2989 * R + 0.587 * G + 0.114 * B`.
""",
            """İkili Dönüşüm

Bu işlem, gri tonlamalı bir resmi ikili bir resme dönüştürür. Her pikselin yoğunluğunu bir eşik değeriyle karşılaştırarak bunu yapar.

Eğer yoğunluk eşik değeri ile büyük ya da eşitse, piksel beyaz (255) olarak ayarlanır; aksi takdirde siyah (0) olarak ayarlanır.
""",
            """Resim Döndürme

a
""",
        ]
        if index == 0:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Lütfen bir işlem seçin.")
        else:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(blue_color)
            self.terminal_codes.append(operation_texts[index])
            
    def upload_image(self):
        file_name = "result\cameraman.jpg"
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(green_color)
            self.terminal_codes.append("Kaynak resim yüklendi.")
    
    def apply_operation(self):
        
        if self.image is None:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Herhangi bir işlem yapmadan önce lütfen bir resim yükleyin.")
            return
        
        operation = self.operation_combo.currentText()
        
        try:
            if operation == 'Select Operation':
                self.terminal_codes.clear()
                self.terminal_codes.setTextColor(red_color)
                self.terminal_codes.append("Lütfen bir işlem seçin.")
                
            elif operation == 'Gray Conversion':
                gray_image = self.convert_to_gray(self.image)
                self.terminal_codes.clear()
                self.download_image(gray_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Gri Dönüşüm işlemi uygulandı.")
                
            elif operation == 'Binary Conversion':
                binary_image = self.convert_to_binary(self.image)
                self.terminal_codes.clear()
                self.download_image(binary_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Binary Dönüşüm işlemi uygulandı.")
                
            elif operation == 'Image Rotation':
                angle, ok = QInputDialog.getDouble(self, 'Image Rotation', 'Enter angle:')
                if ok:
                    self.terminal_codes.clear()
                    if angle < -360 or angle > 360:
                        self.terminal_codes.setTextColor(red_color)
                        self.terminal_codes.append("Please enter a valid angle between -360 and 360.")
                        return
                    else:
                        rotated_image = self.rotate_image(self.image, angle)
                        self.download_image(rotated_image)
                        self.terminal_codes.setTextColor(green_color)
                        if angle in [30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330]:
                            self.terminal_codes.append("Rotation angle:{}\nImage Rotation operation was applied.\nBounding Box: True".format(angle))
                        else:
                            self.terminal_codes.append("Rotation angle:{}\nImage Rotation operation was applied.\nBounding Box: False".format(angle))
            
            elif operation == 'Görüntü Döndürme':
                angle, ok = QInputDialog.getDouble(self, 'Açı', 'Döndürme açısı (derece):', 0, -360, 360, 1)
                if ok:
                    rotated_image = self.rotate_image(self.image, math.radians(angle))
                    self.download_image(rotated_image)
                    self.terminal_codes.setTextColor(green_color)
                    self.terminal_codes.append(f"{angle} derece döndürme uygulandı.")


        except Exception as e:
            self.terminal_codes.append("Error occurred while applying '{}' operation:\n\n{}".format(operation, str(e)))
        finally:
            pass
        
    def download_image(self, image):
        if self.image is None:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Lütfen görseli indirmeden önce bir işlem uygulayınız.")
            return
        
        if self.operation_combo.currentText() == 'Gray Conversion':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
            image.save("result.png", "PNG")
        elif self.operation_combo.currentText() == 'Binary Conversion':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
            image.save("result.png", "PNG")
        elif self.operation_combo.currentText() == 'Image Rotation':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888))
            image.save("result.png", "PNG")

        self.terminal_codes.clear()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Resim result.png olarak indirildi.")
        
######################################## O P E R A S Y O N L A R ########################################
    
    def convert_to_gray(self, image):
        gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray_image[i, j] = int(0.2989 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0])
        return gray_image

    def convert_to_binary(self, image, threshold=128):
        gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray = int(0.2989 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0])
                if gray < threshold:
                    gray_image[i, j] = 0
                else:
                    gray_image[i, j] = 255
        return gray_image

    def rotate_image(self, image, angle):
        if angle in (0, 360, -360):
            return image
        
        if angle not in (30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330, -30, -45, -60, -120, -135, -150, -210, -225, -240, -300, -315, -330):
            bounding_box_image = image
        else:
            bounding_box_image = self.place_image_in_bounding_box(image, angle)
            
        (h, w) = bounding_box_image.shape[:2]
        center = (w // 2, h // 2)
        rotated = np.zeros_like(bounding_box_image)
        
        for i in range(h):
            for j in range(w):
                new_x =  ((j - center[0]) * np.cos(np.deg2rad(angle))) + ((i - center[1]) * np.sin(np.deg2rad(angle))) + center[0]
                new_y = -((j - center[0]) * np.sin(np.deg2rad(angle))) + ((i - center[1]) * np.cos(np.deg2rad(angle))) + center[1]
                if new_x >= 0 and new_x < h and new_y >= 0 and new_y < w:
                    x0 = int(np.floor(new_x))
                    y0 = int(np.floor(new_y))
                    x1 = int(np.ceil(new_x))
                    y1 = int(np.ceil(new_y))
                    if x0 < 0:
                        x0 = 0
                    if y0 < 0:
                        y0 = 0
                    if x1 >= w:
                        x1 = w - 1
                    if y1 >= h:
                        y1 = h - 1
                    a = new_x - x0
                    b = new_y - y0
                    c = 1 - a
                    d = 1 - b
                    for channel in range(3):
                        rotated[i, j, channel] = (c * d * bounding_box_image[y0, x0, channel] + a * d * bounding_box_image[y0, x1, channel] + c * b * bounding_box_image[y1, x0, channel] + a * b * bounding_box_image[y1, x1, channel])
        return rotated

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())