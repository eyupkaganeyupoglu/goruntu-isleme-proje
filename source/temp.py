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
        self.operation_combo.addItem('Gray Conversion')
        self.operation_combo.addItem('Binary Conversion')
        self.operation_combo.addItem('Image Rotation')
        self.operation_combo.addItem('Image Cropping')
        self.operation_combo.addItem('Image Zoom in')
        self.operation_combo.addItem('Image Zoom out')
        self.operation_combo.addItem('RGB to NTSC')
        self.operation_combo.addItem('RGB to YCbCr')
        self.operation_combo.addItem('Histogram Stretching')
        self.operation_combo.addItem('Histogram Widening')
        self.operation_combo.addItem('Arithmetic Operations Addition')
        self.operation_combo.addItem('Arithmetic Operations Division')
        self.operation_combo.addItem('Contrast Increase/Decrease')
        self.operation_combo.addItem('Convolution Operation Mean')
        self.operation_combo.addItem('Thresholding')
        self.operation_combo.addItem('Edge Detection Prewitt')
        self.operation_combo.addItem('Add Noise (Salt & Pepper)')
        self.operation_combo.addItem('Filter Mean')
        self.operation_combo.addItem('Filter Median')
        self.operation_combo.addItem('Filter Unsharp')
        self.operation_combo.addItem('Morphological Operations Dilation')
        self.operation_combo.addItem('Morphological Operations Erosion')
        self.operation_combo.addItem('Morphological Operations Opening')
        self.operation_combo.addItem('Morphological Operations Closing')
        
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
        # file_name = "result\image1.jpg"
        # if file_name:
        #     self.image = cv2.imread(file_name)
        #     self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        #     self.terminal_codes.clear()
        #     self.terminal_codes.setTextColor(green_color)
        #     self.terminal_codes.append("Source image uploaded.")

        try:
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png)')
            if file_name:
                self.image = cv2.imread(file_name)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.terminal_codes.clear()
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Source image uploaded.")
        except Exception as e:
            self.terminal_codes.clear()
            self.terminal_codes.setTextColor(red_color)
            self.terminal_codes.append("Error occurred while uploading image: {}".format(str(e)))
            self.terminal_codes.setTextColor(blue_color)
            self.terminal_codes.append("An error may occur when the path where the image is located does not conform to ASCII standards. Use a path that does not contain Turkish characters.\n\nMake sure that the image you want to upload is in JPG or PNG format.")
        finally:
            pass
    
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

            elif operation == 'RGB to NTSC':
                image = self.convert_rgb_to_ntsc(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("RGB to NTSC operation was applied.")

            elif operation == 'RGB to YCbCr':
                image = self.convert_rgb_to_ycbcr(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("RGB to YCbCr operation was applied.")

            elif operation == 'Histogram Stretching':
                stretched_image = self.histogram_stretching(self.image)
                self.terminal_codes.clear()
                self.download_image(stretched_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Histogram Stretching operation was applied.")

            elif operation == 'Histogram Widening':
                widened_image = self.histogram_widening(self.image)
                self.terminal_codes.clear()
                self.download_image(widened_image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Histogram Widening operation was applied.")

            elif operation == 'Arithmetic Operations Addition':
                try:
                    file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png)')
                    if file_name:
                        image2 = cv2.imread(file_name)
                        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                        image = self.arithmetic_operations_addition(self.image, image2)
                        self.terminal_codes.clear()
                        self.terminal_codes.setTextColor(green_color)
                        self.terminal_codes.append("Addition operation applied successfully.")
                        self.download_image(image)
                    else:
                        raise ValueError("No image file selected.")
                except Exception as e:
                    self.terminal_codes.clear()
                    self.terminal_codes.setTextColor(red_color)
                    self.terminal_codes.append(f"Error occurred while uploading image: {str(e)}")
                    self.terminal_codes.setTextColor(blue_color)
                    self.terminal_codes.append("An error may occur when the path where the image is located does not conform to ASCII standards. Use a path that does not contain Turkish characters.\n\nMake sure that the image you want to upload is in JPG or PNG format.")
                finally:
                    pass

            elif operation == 'Arithmetic Operations Division':
                try:
                    file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png)')
                    if file_name:
                        image2 = cv2.imread(file_name)
                        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                        image = self.arithmetic_operations_division(self.image, image2)
                        self.terminal_codes.clear()
                        self.terminal_codes.setTextColor(green_color)
                        self.terminal_codes.append("Division operation applied successfully.")
                        self.download_image(image)
                    else:
                        raise ValueError("No image file selected.")
                except Exception as e:
                    self.terminal_codes.clear()
                    self.terminal_codes.setTextColor(red_color)
                    self.terminal_codes.append(f"Error occurred while uploading image: {str(e)}")
                    self.terminal_codes.setTextColor(blue_color)
                    self.terminal_codes.append("An error may occur when the path where the image is located does not conform to ASCII standards. Use a path that does not contain Turkish characters.\n\nMake sure that the image you want to upload is in JPG or PNG format.")
                finally:
                    pass

            elif operation == 'Contrast Increase/Decrease':
                contrast_value, ok = QInputDialog.getDouble(self, 'Contrast Count', 'Enter contrast count (default=1,0):')
                if ok:
                    enhanced_image = self.contrast_increase_decrease(self.image, alpha=contrast_value, beta=0)
                    self.terminal_codes.clear()
                    self.download_image(enhanced_image)
                    self.terminal_codes.setTextColor(green_color)
                    self.terminal_codes.append(f"Contrast boosting applied with alpha={contrast_value} successfully.")

            elif operation == 'Convolution Operation Mean':
                image = self.convolution_operation_mean(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Convolution (Mean Filter) applied successfully.")

            elif operation == 'Thresholding':
                threshold_value, ok = QInputDialog.getInt(self, 'Threshold Value', 'Enter threshold value (0-255, default=127):')
                if ok:
                    image = self.thresholding(self.image, threshold=threshold_value)
                    self.terminal_codes.clear()
                    self.download_image(image)
                    self.terminal_codes.setTextColor(green_color)
                    self.terminal_codes.append(f"Thresholding applied successfully with threshold={threshold_value}.")

            elif operation == 'Edge Detection Prewitt':
                image = self.edge_detection_prewitt(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append("Edge Detection Prewitt operation was applied.")

            elif operation == 'Add Noise (Salt & Pepper)':
                image = self.add_noise_salt_and_pepper(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Salt & Pepper noise added.")

            elif operation == 'Filter Mean':
                image = self.filter_mean(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Mean filter applied successfully.")

            elif operation == 'Filter Median':
                image = self.filter_median(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Median filter applied successfully.")

            elif operation == 'Filter Unsharp':
                image = self.filter_unsharp(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Unsharp filter applied successfully.")

            elif operation == 'Morphological Operations Dilation':
                image = self.morphological_operations_dilation(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Dilation operation applied successfully.")

            elif operation == 'Morphological Operations Erosion':
                image = self.morphological_operations_erosion(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Erosion operation applied successfully.")

            elif operation == 'Morphological Operations Opening':
                image = self.morphological_operations_opening(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Opening operation applied successfully.")

            elif operation == 'Morphological Operations Closing':
                image = self.morphological_operations_closing(self.image)
                self.terminal_codes.clear()
                self.download_image(image)
                self.terminal_codes.setTextColor(green_color)
                self.terminal_codes.append(f"Closing operation applied successfully.")

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
        elif self.operation_combo.currentText() == 'RGB to NTSC':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'RGB to YCbCr':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Histogram Stretching':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Histogram Widening':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Arithmetic Operations Addition':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Arithmetic Operations Division':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Contrast Increase/Decrease':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Convolution Operation Mean':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Thresholding':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Edge Detection Prewitt':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Add Noise (Salt & Pepper)':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Filter Mean':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Filter Median':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Filter Unsharp':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Morphological Operations Dilation':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Morphological Operations Erosion':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Morphological Operations Opening':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")
        elif self.operation_combo.currentText() == 'Morphological Operations Closing':
            image = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8))
            image.save(r"result\result.png", "PNG")

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
    
    def convert_rgb_to_ntsc(self, image):
        image = image.astype(np.float64)
        
        transformation_matrix = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.275, -0.321],
            [0.212, -0.523, 0.311]
        ])
        
        yiq_image = np.dot(image[..., :3], transformation_matrix.T)

        yiq_image = np.clip(yiq_image, 0, 255).astype(np.uint8)
        
        return yiq_image
    
    def convert_rgb_to_ycbcr(self, image):
        if image.dtype == np.uint8:
            delta = 128
        elif image.dtype == np.uint16:
            delta = 32768
        else:
            delta = 0.5

        height, width, _ = image.shape
        ycbcr_image = np.zeros_like(image, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                R, G, B = image[y, x]

                Y = 0.299 * R + 0.587 * G + 0.114 * B
                Cr = (R - Y) * 0.713 + delta
                Cb = (B - Y) * 0.564 + delta

                ycbcr_image[y, x] = [Y, Cb, Cr]

        if delta == 128:
            ycbcr_image = np.clip(ycbcr_image, 0, 255).astype(np.uint8)
        elif delta == 0.5:
            ycbcr_image = np.clip(ycbcr_image, 0, 1)
        elif delta == 32768:
            ycbcr_image = np.clip(ycbcr_image, 0, 65535).astype(np.uint16)

        return ycbcr_image
    
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
        if len(image.shape) == 3:
            gray = self.convert_to_gray(image)
        else:
            gray = image

        min_val = np.min(gray)
        max_val = np.max(gray)

        if max_val - min_val == 0:
            return gray.copy()

        widened = ((gray - min_val) / (max_val - min_val)) * 255
        widened = np.clip(widened, 0, 255).astype(np.uint8)

        return widened
    
    def arithmetic_operations_addition(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must be the same size for addition.")
        
        height, width, channels = image1.shape
        result = np.zeros((height, width, channels), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    result[i, j, c] = (int(image1[i, j, c]) + int(image2[i, j, c])) // 2

        return result

    def arithmetic_operations_division(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must be the same size for division.")
        
        height, width, channels = image1.shape
        result = np.zeros((height, width, channels), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    denominator = int(image2[i, j, c])
                    if int(image2[i, j, c]) == 0:
                        result[i, j, c] = 255
                    else:
                        result[i, j, c] = min(int(image1[i, j, c]) // denominator, 255)

        return result

    def contrast_increase_decrease(self, image, alpha=1.0, beta=0):
        image_float = image.astype(np.float32)
        result = alpha * (image_float - 128) + 128 + beta
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def convolution_operation_mean(self, image):
        height, width, channels = image.shape
        padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        result = np.zeros_like(image, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    neighborhood = padded_image[y:y+3, x:x+3, c]
                    mean_value = np.mean(neighborhood)
                    result[y, x, c] = int(mean_value)

        return result

    def thresholding(self, image, threshold=127):
        if len(image.shape) == 3:
            gray = self.convert_to_gray(image)
        else:
            gray = self.image

        thresholded_image = np.zeros_like(gray, dtype=np.uint8)

        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                if gray[y, x] > threshold:
                    thresholded_image[y, x] = 255
                else:
                    thresholded_image[y, x] = 0

        return thresholded_image

    def edge_detection_prewitt(self, image):
        if len(image.shape) == 3:
            gray = self.convert_to_gray(image)
        else:
            gray = image

        height, width = gray.shape
        result = np.zeros_like(gray)

        kernel_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        kernel_y = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

        padded = np.pad(gray, ((1, 1), (1, 1)), mode='reflect')

        for i in range(1, height + 1):
            for j in range(1, width + 1):
                region = padded[i - 1:i + 2, j - 1:j + 2]
                gx = np.sum(kernel_x * region)
                gy = np.sum(kernel_y * region)
                gradient = min(int(np.sqrt(gx ** 2 + gy ** 2)), 255)
                result[i - 1, j - 1] = gradient

        return result

    def add_noise_salt_and_pepper(self, image):
        amount=0.05

        output = np.copy(image)
        row, col = output.shape[:2]
        num_pixels = row * col

        num_salt = int(num_pixels * amount / 2)
        num_pepper = int(num_pixels * amount / 2)

        for _ in range(num_salt):
            i = np.random.randint(0, row)
            j = np.random.randint(0, col)
            if len(output.shape) == 2:
                output[i, j] = 255
            else:
                output[i, j] = [255, 255, 255]

        for _ in range(num_pepper):
            i = np.random.randint(0, row)
            j = np.random.randint(0, col)
            if len(output.shape) == 2:
                output[i, j] = 0
            else:
                output[i, j] = [0, 0, 0]

        return output

    def filter_mean(self, image):
        image = self.add_noise_salt_and_pepper(image)

        height, width, channels = image.shape
        padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        result = np.copy(image)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    pixel_value = image[y, x, c]
                    if pixel_value == 0 or pixel_value == 255:
                        neighborhood = padded_image[y:y+3, x:x+3, c]
                        valid_neighbors = neighborhood[(neighborhood != 0) & (neighborhood != 255)]

                        if valid_neighbors.size > 0:
                            mean_value = np.mean(valid_neighbors)
                        else:
                            mean_value = np.mean(neighborhood)

                        result[y, x, c] = int(mean_value)
        return result
    
    def filter_median(self, image):
        image = self.add_noise_salt_and_pepper(image)

        height, width, channels = image.shape
        padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        result = np.copy(image)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    pixel_value = image[y, x, c]
                    if pixel_value == 0 or pixel_value == 255:
                        neighborhood = padded_image[y:y+3, x:x+3, c]
                        valid_neighbors = neighborhood[(neighborhood != 0) & (neighborhood != 255)]

                        if valid_neighbors.size > 0:
                            median_value = np.median(valid_neighbors)
                        else:
                            median_value = np.median(neighborhood)

                        result[y, x, c] = int(median_value)
        return result

    def filter_unsharp(self, image, amount=1.0):
        if len(image.shape) == 3:
            image_gray = self.convert_to_gray(image)
        else:
            image_gray = image

        image_gray = image_gray.astype(np.float32)

        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        padded = np.pad(image_gray, ((1, 1), (1, 1)), mode='reflect')
        blurred = np.zeros_like(image_gray)

        height, width = image_gray.shape

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                region = padded[y-1:y+2, x-1:x+2]
                blurred[y-1, x-1] = np.sum(region * kernel)

        mask = image_gray - blurred

        sharpened = image_gray + amount * mask

        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def morphological_operations_dilation(self, image):
        if len(image.shape) == 3:
            binary_image = self.convert_to_binary(image)
        else:
            binary_image = image

        height, width = binary_image.shape

        dilated_image = np.copy(binary_image)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if binary_image[y, x] == 255:
                    dilated_image[y-1:y+2, x-1:x+2] = 255

        return dilated_image

    def morphological_operations_erosion(self, image):
        if len(image.shape) == 3:
            binary_image = self.convert_to_binary(image)
        else:
            binary_image = image

        height, width = binary_image.shape

        eroded_image = np.copy(binary_image)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if binary_image[y, x] == 0:
                    eroded_image[y-1:y+2, x-1:x+2] = 0

        return eroded_image

    def morphological_operations_opening(self, image):
        if len(image.shape) == 3:
            binary_image = self.convert_to_binary(image)
        else:
            binary_image = image

        eroded_image = self.morphological_operations_erosion(binary_image)

        opened_image = self.morphological_operations_dilation(eroded_image)

        return opened_image

    def morphological_operations_closing(self, image):
        if len(image.shape) == 3:
            binary_image = self.convert_to_binary(image)
        else:
            binary_image = image

        dilated_image = self.morphological_operations_dilation(binary_image)

        closed_image = self.morphological_operations_erosion(dilated_image)

        return closed_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())