# Python version:  3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
# OpenCV version:  4.9.0
# Numpy version:   1.26.4
# PyQt5 version:   5.15.10

# DİKKAT: cv2 türkçe karakterlere karşı sorun çıkarıyor. Bu yüzden uygulamayı ascii'ye uygun karakterler içeren path'da çalıştırdığınıza emin olun...

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
        self.operation_combo.addItem('Binary Conversion')
        self.operation_combo.addItem('Image Rotation')
        self.operation_combo.addItem('Image Cropping')
        self.operation_combo.addItem('Image Zoom in')
        self.operation_combo.addItem('Image Zoom out')
        self.operation_combo.addItem('RGB to HSV')
        self.operation_combo.addItem('RGB to YCbCr')
        # self.operation_combo.addItem('Histogram Stretching/Widening')
        # self.operation_combo.addItem('Arithmetic Operations (Addition, Division)')
        # self.operation_combo.addItem('Contrast Enhancement')
        # self.operation_combo.addItem('Convolution Operation (Mean)')
        # self.operation_combo.addItem('Thresholding Operations (Single Thresholding)')
        # self.operation_combo.addItem('Edge Detection Algorithms (Prewitt)')
        # self.operation_combo.addItem('Noise Removal (Salt & Pepper)')
        # self.operation_combo.addItem('Apply Filter (Unsharp)')
        # self.operation_combo.addItem('Morphological Operations (Expansion, Abrasion, Opening, Closing)')
        
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
        # DİKKAT: operation_texts'da operasyonların ne işe yaradıklarını ve nasıl çalıştıklarını toplam 2 cümle ile açıkla. Matematiksel formülleri de ekle.
        operation_texts = [
            "İşlem Seç",
            """Gri Dönüşüm

Bu işlem, renkli bir görüntüyü gri tonlamalı bir görüntüye dönüştürür.

Her pikselin kırmızı, yeşil ve mavi kanallarının ağırlıklı toplamını hesaplayarak bunu yapar. Kullanılan formül:

    `0.2989 * R + 0.587 * G + 0.114 * B`.
""",
            """İkili Dönüşüm

Bu işlem, gri tonlamalı bir görüntüyü ikili (siyah-beyaz) bir görüntüye dönüştürür. Bunu, her pikselin yoğunluğunu bir eşik değeriyle karşılaştırarak yapar.

Yoğunluk eşik değerine eşit veya büyükse piksel beyaz (255) yapılır; değilse siyah (0) yapılır.
""",
            """Görüntü Döndürme

Bu işlem, bir görüntüyü istenilen herhangi bir açıyla döndürmemize olanak tanır.

bounding_box
Görüntünün yükseklik, genişlik ve döndürme açısı değerleri kullanılarak çerçevenin boyutu hesaplanır ve bu boyuta göre sıfırlarla dolu bir matris oluşturulur:

    new_w = ceil(w * abs(sin(rad)) + h * abs(cos(rad)))
    new_h = ceil(h * abs(sin(rad)) + w * abs(cos(rad)))

place_image_in_bounding_box
Görüntünün yükseklik ve genişlik bilgileri ile çerçevenin boyutları kullanılarak çerçevenin merkez noktası hesaplanır. Daha sonra, çerçevenin merkez noktasından görüntünün yükseklik ve genişlik değerleri çıkarılarak görüntünün çerçeve içindeki sol üst pikselinin koordinatları elde edilir. Bu pikselden başlayarak görüntü çerçeve içine piksel piksel yerleştirilir.

rotate_image
Her piksel, görüntünün çerçeve içine yerleştirilmesiyle elde edilen merkez koordinatları kullanılarak döndürülür:

    x' = x * cos(angle) - y * sin(angle)
    y' = x * sin(angle) + y * cos(angle) 

Not: new_x ve new_y döndürme matris formülü, koordinat sistemindeki (0,0) orijin noktasına göre döndürme yapar. (0,0) noktasını elde etmek için, görüntünün çerçeve içine yerleştirilmesiyle elde edilen merkez noktası, i ve j’den x, y değerleri çıkarılarak döndürülür. Daha sonra görüntüyü çerçevenin ortasına getirmek için, çerçeveye yerleştirilen görüntünün merkez koordinatları eklenir.

Döndürülmüş pikselin görüntüdeki üst sol (x0, y0) ve alt sağ (x1, y1) koordinatları bulunur. Daha sonra bu görüntünün etrafındaki 4 pikselin (a, b, c, d) ağırlıkları hesaplanır:

    (x0, y0) pikseli için ağırlık c * d
    (x1, y0) pikseli için ağırlık a * d
    (x0, y1) pikseli için ağırlık c * b
    (x1, y1) pikseli için ağırlık a * b

Daha sonra bu 4 çevre pikselin ağırlıkları kullanılarak her pikselin RGB değeri Bilineer Enterpolasyon yöntemiyle hesaplanır.

    f(x, y) ≈ (1 - a)(1 - b) f(x0, y0) + a(1 - b) f(x1, y0) + (1 - a)b f(x0, y1) + ab f(x1, y1)
""",
            """Görüntü Kırpma

Bu işlem, bir görüntünün istenilen iki koordinat arasındaki kısmını kırpmak için kullanılır.

Bu işlem, iki koordinattan birini sol üst köşe ve diğerini sağ alt köşe kabul ederek bir dörtgen çizer ve bu kısmı döndürür.
""",
            """Görüntü Yakınlaştırma

Bu işlem, görüntünün En Yakın Komşu Enterpolasyonu ile yakınlaştırılmasını sağlar.
""",
            """Görüntü Uzaklaştırma

Bu işlem, görüntünün En Yakın Komşu Enterpolasyonu ile uzaklaştırılmasını sağlar.
""",
            """RGB'den HSV'ye Dönüşüm

Bu işlem, RGB (Kırmızı, Yeşil, Mavi) renk uzayındaki bir görüntüyü HSV (Ton, Doygunluk, Parlaklık) renk uzayına dönüştürmek için kullanılır.

Dönüşüm şu adımlarla gerçekleştirilir:

1. R, G, B renk değerleri 0 ile 1 arasında olacak şekilde normalize edilir (örneğin, 255 üzerinden gelen bir değer 255'e bölünür).

2. R, G ve B değerleri arasındaki en büyük (maksimum) ve en küçük (minimum) değerler bulunur.

3. Bu iki değer arasındaki fark (delta) hesaplanır.

4. Ton (Hue) değeri şu kurallara göre hesaplanır:
   - En büyük değer R ise, Ton = (G - B) / delta
   - En büyük değer G ise, Ton = (B - R) / delta + 2
   - En büyük değer B ise, Ton = (R - G) / delta + 4
   - Sonuç 60 ile çarpılır. Eğer delta sıfırsa, Ton değeri sıfır alınır.

5. Doygunluk (Saturation) şu şekilde hesaplanır:
   - Eğer maksimum değer sıfırsa, doygunluk sıfırdır.
   - Aksi halde, doygunluk = delta / maksimum değer

6. Parlaklık (Value) doğrudan maksimum değer olarak alınır.

Sonuçta, Ton 0 ile 360 derece arasında bir açı olarak ifade edilir.
Doygunluk ve Parlaklık ise 0 ile 1 arasında değerler alır.
""",
            """RGB'den YCbCr'ye Dönüşüm
            
Bu işlem, RGB renk uzayındaki bir görüntüyü YCbCr (Parlaklık ve renk farkı bileşenleri) renk uzayına dönüştürmek için kullanılır. Dönüşüm için şu işlemler yapılır:

1. R, G, B değerleri [0, 255] aralığında kabul edilir.

2. Y, Cb ve Cr bileşenleri aşağıdaki dönüşüm formülleri ile hesaplanır:

    Y  =  0.299 × R + 0.587 × G + 0.114 × B
    Cb = -0.168736 × R - 0.331264 × G + 0.5 × B + 128
    Cr =  0.5 × R - 0.418688 × G - 0.081312 × B + 128

3. Y bileşeni parlaklığı (luminance), Cb ve Cr bileşenleri ise mavi ve kırmızıya olan renk farklarını temsil eder.

Sonuçta, Y [0, 255], Cb ve Cr [0, 255] aralığındadır.
""",
            # "Histogram Stretching/Widening",
            # "Arithmetic Operations (Addition, Division)",
            # "Contrast Enhancement",
            # "Convolution Operation (Mean)",
            # "Thresholding Operations (Single Thresholding)",
            # "Edge Detection Algorithms (Prewitt)",
            # "Noise Removal (Salt & Pepper)",
            # "Apply Filter (Unsharp)",
            # "Morphological Operations (Expansion, Abrasion, Opening, Closing)"
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
        # NOT (BİLGİLENDİRME): Temporary code for testing
        file_name = "result\cameraman.jpg"
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(green_color)
            self.terminal_codes.append("Source image uploaded.")
        # NOT (BİLGİLENDİRME): Actual code
        # try:
        #     file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png)')
        #     if file_name:
        #         self.image = cv2.imread(file_name)
        #         self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        #         self.terminal_codes.clear()
        #         self.terminal_codes.setTextColor(green_color)
        #         self.terminal_codes.append("Source image uploaded.")
        # except Exception as e:
        #     self.terminal_codes.clear()
        #     self.terminal_codes.setTextColor(red_color)
        #     self.terminal_codes.append("Error occurred while uploading image: {}".format(str(e)))
        #     self.terminal_codes.setTextColor(blue_color)
        #     self.terminal_codes.append("An error may occur when the path where the image is located does not conform to ASCII standards. Use a path that does not contain Turkish characters.\n\nMake sure that the image you want to upload is in JPG or PNG format.")
        # finally:
        #     pass

    # TODO: Birinci grubun ödevi olan bütün operasyonları ve bunların fonksiyonlarını yazdıktan sonra GitHub için ikinci ve üçüncü grubun ödevi olan bütün operasyonları ve bunların fonksiyonları da ekle.
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

            elif operation == 'Binary Conversion':
                binary_image = self.convert_to_binary(self.image)
                self.terminal_codes.clear()
                self.download_image(binary_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Binary Conversion operation was applied.")

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
                        self.terminal_codes.append("Rotation angle:{}\nImage Rotation operation was applied.".format(angle))

            elif operation == 'Image Cropping':
                x1, ok = QInputDialog.getInt(self, 'Image Cropping', f'Enter x1 ({self.image.shape[1]}):')
                if ok:
                    y1, ok = QInputDialog.getInt(self, 'Image Cropping', f'Enter y1 ({self.image.shape[0]}):')
                    if ok:
                        x2, ok = QInputDialog.getInt(self, 'Image Cropping', f'Enter x2 ({self.image.shape[1]}):')
                        if ok:
                            y2, ok = QInputDialog.getInt(self, 'Image Cropping', f'Enter y2 ({self.image.shape[0]}):')
                            if ok:
                                cropped_image = self.crop_image(self.image, x1, y1, x2, y2)
                                self.terminal_codes.clear()
                                if cropped_image is not None:
                                    self.download_image(cropped_image)
                                    self.terminal_codes.setTextColor(green_color)
                                    self.terminal_codes.append("Image Cropping operation was applied.")
                                else:
                                    if  x1 > self.image.shape[1] or y1 > self.image.shape[0] or x2 > self.image.shape[1] or y2 > self.image.shape[0]:
                                        self.terminal_codes.setTextColor(red_color)
                                        self.terminal_codes.append(f"Coordinates are outside the image boundaries. Please enter coordinates in the {self.image.shape[1]} - {self.image.shape[0]} range.")
                                    if x1 == x2 or y1 == y2:
                                        self.terminal_codes.setTextColor(red_color)
                                        self.terminal_codes.append("Coordinates must specify an area. So x1 and x2 or y1 and y2 cannot be equal.")
                                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                                        self.terminal_codes.setTextColor(red_color)
                                        self.terminal_codes.append("Coordinates cannot be negative.")

            elif operation == 'Image Zoom in':
                zoom_value, ok = QInputDialog.getItem(self, "Zoom Value", "Select Zoom Value", ['2','4'], 0, False)
                if ok:
                    self.terminal_codes.clear()
                    self.terminal_codes.setTextColor(blue_color)
                    zoomed = self.zoom_in_image(self.image, int(zoom_value))
                    self.download_image(zoomed)
                    self.terminal_codes.append("Image zoomed in with a ratio of " + zoom_value + ".")

            elif operation == 'Image Zoom out':
                zoom_value, ok = QInputDialog.getItem(self, "Zoom Value", "Select Zoom Value", ['2','4'], 0, False)
                if ok:
                    self.terminal_codes.clear()
                    self.terminal_codes.setTextColor(blue_color)
                    zoomed = self.zoom_out_image(self.image, int(zoom_value))
                    self.download_image(zoomed)
                    self.terminal_codes.append("Image zoomed out with a ratio of " + zoom_value + ".")

            elif operation == 'RGB to HSV':
                hsv_image = self.convert_rgb_to_hsv(self.image)
                self.terminal_codes.clear()
                self.download_image(hsv_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("RGB to HSV operation was applied.")

            elif operation == 'RGB to YCbCr':
                ycbcr_image = self.convert_rgb_to_ycbcr(self.image)
                self.terminal_codes.clear()
                self.download_image(ycbcr_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("RGB to YCbCr operation was applied.")

            # elif operation == 'Histogram Stretching/Widening':
            #     choice, ok = QInputDialog.getItem(self, 'Histogram Stretching/Widening', 'Select operation:', ('Histogram Stretching', 'Histogram Widening'))
            #     if ok:
            #         value, ok = QInputDialog.getInt(self, 'Contrast Adjustment', 'Enter contrast value:')
            #         if ok:
            #             if choice == 'Histogram Stretching':
            #                 self.result_image_label.setPixmap(self.stretch_histogram(self.image, value/10))
            #                 self.terminal_codes.append("Histogram Stretching operation was applied.")
            #             elif choice == 'Histogram Widening':
            #                 self.result_image_label.setPixmap(self.widen_histogram(self.image, value/10))
            #                 self.terminal_codes.append("Histogram Widening operation was applied.")
                
            # elif operation == 'Arithmetic Operations (Addition, Division)':
            #     file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png)')
            #     if file_name:
            #         image2 = cv2.imread(file_name)
            #         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            #         op, ok = QInputDialog.getItem(self, 'Arithmetic Operations', 'Select operation:', ('Addition', 'Division'))
            #         if ok:
            #             if op == 'Addition':
            #                 result = cv2.add(self.image, image2)
            #             elif op == 'Division':
            #                 result = cv2.divide(self.image, image2)
            #             self.result_image_label.setPixmap(QPixmap.fromImage(QImage(result, result.shape[1], result.shape[0], QImage.Format_RGB888)))
            #             self.terminal_codes.append("Arithmetic Operations operation was applied.")

            # elif operation == 'Contrast Enhancement':
            #     self.result_image_label.setPixmap(self.enhance_contrast(self.image))
                
            # elif operation == 'Convolution Operation (Mean)':
            #     self.result_image_label.setPixmap(self.mean_filter(self.image))
                
            # elif operation == 'Thresholding Operations (Single Thresholding)':
            #     threshold, ok = QInputDialog.getInt(self, 'Thresholding Operations', 'Enter threshold:')
            #     if ok:
            #         self.result_image_label.setPixmap(self.single_thresholding(self.image, threshold))
                
            # elif operation == 'Edge Detection Algorithms (Prewitt)':
            #     self.result_image_label.setPixmap(self.prewitt_edge_detection(self.image))
                
            # elif operation == 'Noise Removal (Salt & Pepper)':
            #     self.result_image_label.setPixmap(self.salt_pepper_noise_removal(self.image))
                
            # elif operation == 'ApplyFilter (Unsharp)':
            #     self.result_image_label.setPixmap(self.unsharp_filter(self.image))
                
            # elif operation == 'Morphological Operations (Expansion, Abrasion, Opening, Closing)':
            #     self.result_image_label.setPixmap(self.morphological_operation(self.image))
        
        # DİKKAT: Operation'ları yazmayı bitirdikten sonra o operation'a özel hatalar olursa bu hatalar için özel error mesajları hazırla.
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
        
        # DİKKAT: Diğer operasyonlar için de bu sorgulamayı yap.
        # TODO: Hepsi bitince aynı formattakileri birleştir.
        if self.operation_combo.currentText() == 'Gray Conversion':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Binary Conversion':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Image Rotation':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Image Cropping':
            cv2.imwrite(r"result\result.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif self.operation_combo.currentText() == 'Image Zoom in':
            image = QPixmap.fromImage(QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Image Zoom out':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'RGB to HSV':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'RGB to YCbCr':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")

        self.terminal_codes.clear()
        self.terminal_codes.setTextColor(green_color)
        self.terminal_codes.append("Image downloaded as result.png.")
        
######################################## O P E R A S Y O N L A R ########################################
    
    # TODO: Fonksiyonları okul PDF'lerinden araştır.
        # TODO: Proje dağılım belgesinden doğru fonksiyonu mu araştırıyorsun kontrol et.
        # TODO: PDF'lerden ve internetten araştırarak fonksiyonları matematiksel olarak ifade edebil.
        # TODO: Programın donmadığını, çalıştığını kullanıcıya görsel bilgi olarak sunabil diye fonksiyonlar çalışırken ürettikleri değerleri terminal_codes'ya yazdır.
        # TODO: Birinci grubun ödevi olan bütün operasyonları ve bunların fonksiyonlarını yazdıktan sonra GitHub için ikinci ve üçüncü grubun ödevi olan bütün operasyonları ve bunların fonksiyonları da ekle.
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

    def bounding_box(self, image, angle):
        (h, w) = image.shape[:2]
        
        rad = np.deg2rad(angle)
        new_w = int(np.ceil(w * np.abs(np.sin(rad)) + h * np.abs(np.cos(rad))))
        new_h = int(np.ceil(h * np.abs(np.sin(rad)) + w * np.abs(np.cos(rad))))
        rotated = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        return rotated

    def place_image_in_bounding_box(self, image, angle):
        rotated = self.bounding_box(image, angle)
        
        (h, w) = image.shape[:2]
        (new_h, new_w) = rotated.shape[:2]
        center_x = new_w // 2
        center_y = new_h // 2

        start_x = center_x - w // 2
        start_y = center_y - h // 2

        rotated[start_y:start_y+h, start_x:start_x+w] = image
        
        return rotated

    # TODO: Bounding box olayını her açıda çalışacak şekilde yapamadım neden bilmiyorum. Hocaya sor.
    def rotate_image(self, image, angle):
        if angle in (0, 360, -360):
            return image
        
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
    
    def crop_image(self, image, x1, y1, x2, y2):
        if (x1 < 0 or y1 < 0 or x1 > image.shape[1] or y1 > image.shape[0] or x2 < 0 or y2 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x1 == x2 or y1 == y2):
            return None
        
        x1_, y1_ = min(x1, x2), min(y1, y2)
        x2_, y2_ = max(x1, x2), max(y1, y2)

        cropped = image[y1_:y2_, x1_:x2_]
        return cropped

    # DESCRIPTION: https://i.stack.imgur.com/smBqi.png
    # DESCRIPTION: https://i.stack.imgur.com/5Okih.png
    def zoom_in_image(self, image, zoom_value):
        h, w = image.shape[:2]
        new_h = h * zoom_value
        new_w = w * zoom_value
        zoomed = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                original_i = i / zoom_value
                original_j = j / zoom_value
                nearest_i = int(round(original_i))
                nearest_j = int(round(original_j))
                nearest_i = max(0, min(nearest_i, h-1))
                nearest_j = max(0, min(nearest_j, w-1))
                zoomed[i, j] = image[nearest_i, nearest_j]
        return zoomed
    
    def zoom_out_image(self, image, zoom_value):
        h, w = image.shape[:2]
        new_h = h // zoom_value
        new_w = w // zoom_value
        if new_h == 0 or new_w == 0:
            return image.copy()
        zoomed = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                start_i = i * zoom_value
                end_i = start_i + zoom_value
                start_j = j * zoom_value
                end_j = start_j + zoom_value
                block = image[start_i:end_i, start_j:end_j]
                avg = np.mean(block, axis=(0, 1)).astype(np.uint8)
                zoomed[i, j] = avg
        return zoomed
    
    def convert_rgb_to_hsv(self, image):
        hsv_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j] / 255.0
                cmax = max(r, g, b)
                cmin = min(r, g, b)
                delta = cmax - cmin

                if delta == 0:
                    h = 0
                elif cmax == r:
                    h = (60 * ((g - b) / delta) + 360) % 360
                elif cmax == g:
                    h = (60 * ((b - r) / delta) + 120) % 360
                else:
                    h = (60 * ((r - g) / delta) + 240) % 360

                s = 0 if cmax == 0 else delta / cmax
                v = cmax

                hsv_image[i, j] = [int(h / 2), int(s * 255), int(v * 255)]
        return hsv_image

    
    def convert_rgb_to_ycbcr(self, image):
        ycbcr_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j]
                y  =  0.299 * r + 0.587 * g + 0.114 * b
                cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
                cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 128
                ycbcr_image[i, j] = [int(y), int(cb), int(cr)]
        return ycbcr_image

    # def widen_histogram(self, image, level):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     img_eq = cv2.equalizeHist(img)
    #     img_stretched = cv2.addWeighted(img, level, img_eq, 1 - level, 0)
    #     return QPixmap.fromImage(QImage(img_stretched, img_stretched.shape[1], img_stretched.shape[0], QImage.Format_Grayscale8))

    # def stretch_histogram(self, image, level):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     img_eq = cv2.equalizeHist(img)
    #     img_widen = cv2.addWeighted(img, 1 - level, img_eq, level, 0)
    #     return QPixmap.fromImage(QImage(img_widen, img_widen.shape[1], img_widen.shape[0], QImage.Format_Grayscale8))

    # def add_images(self, image1, image2):
    #         return QPixmap.fromImage(QImage(cv2.add(image1, image2), image1.shape[1], image1.shape[0], QImage.Format_RGB888))

    # def divide_images(self, image1, image2):
    #     return QPixmap.fromImage(QImage(cv2.divide(image1, image2), image1.shape[1], image1.shape[0], QImage.Format_RGB888))

    # def enhance_contrast(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #     img = clahe.apply(img)
    #     return QPixmap.fromImage(QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8))

    # def mean_filter(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     kernel = np.ones((3,3),np.float32)/9
    #     img = cv2.filter2D(img, -1, kernel)
    #     return QPixmap.fromImage(QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8))

    # def single_thresholding(self, image, threshold):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    #     return QPixmap.fromImage(QImage(binary, binary.shape[1], binary.shape[0], QImage.Format_Grayscale8))

    # def prewitt_edge_detection(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    #     sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    #     abs_sobel_x = np.absolute(sobel_x)
    #     abs_sobel_y = np.absolute(sobel_y)
    #     edges = np.uint8(np.sqrt(abs_sobel_x**2 + abs_sobel_y**2))
    #     return QPixmap.fromImage(QImage(edges, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8))

    # def salt_pepper_noise_removal(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     noise = np.random.randint(0, 255, (img.shape[0], img.shape[1]))
    #     img[noise > 240] = 255
    #     img[noise < 10] = 0
    #     return QPixmap.fromImage(QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8))

    # def unsharp_filter(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     blurred = cv2.GaussianBlur(img, (5, 5), 0)
    #     sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    #     return QPixmap.fromImage(QImage(sharpened, sharpened.shape[1], sharpened.shape[0], QImage.Format_Grayscale8))

    # def morphological_operation(self, image):
    #     img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     kernel = np.ones((5,5),np.uint8)
    #     opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #     closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #     return QPixmap.fromImage(QImage(opening, opening.shape[1], opening.shape[0], QImage.Format_Grayscale8))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
    
# TODO (End): Kodda gerekli yerlere comment ekleyerek kodların nasıl çalıştıklarını açıkla.
# TODO (End): download_image fonksiyonunda QImage.Format'ı her fonksiyon için farklı bir sorgu olacak şekilde ayarladın. Bunları aynı QImage.Format içerenleri birleştir.
# TODO (End): Uzun sürecek işlemleri terminal_codes'da kırmızı ile belirt ki program dondu sanıp kapamasınlar. 