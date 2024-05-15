from hashlib import new
from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2, time, sys
import numpy as np
import random as rnd
import psutil

class ThreadClass(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FPS = pyqtSignal(int)
    
    def run(self):
        global camIndex
        Capture = cv2.VideoCapture(camIndex)
        Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.ThreadActive = True
        prev_frame_time = 0
        new_frame_time = 0
        while self.ThreadActive:
            ret, frame_cap = Capture.read()
            if not ret:
                continue
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            self.ImageUpdate.emit(frame_cap)
            self.FPS.emit(fps)
    
    def stop(self):
        self.ThreadActive = False
        self.quit()
        
class boardInfoClass(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)
    
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            self.cpu.emit(cpu)
            self.ram.emit((ram.total, ram.used, ram.percent))

    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("Main_Window.ui", self)
        self.online_cam = QCameraInfo.availableCameras()
        self.camlist.addItems([c.description() for c in self.online_cam])
        self.btn_start.clicked.connect(self.StartWebCam)
        self.btn_stop.clicked.connect(self.StopWebcam)
        self.btn_stop.setEnabled(False)

        self.resource_usage = boardInfoClass()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.getCPU_usage)
        self.resource_usage.ram.connect(self.getRAM_usage)
        
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 
        
        self.btn_reset_rgb.clicked.connect(self.reset_rgb)
        self.slider_red.valueChanged.connect(self.get_ColorRED)
        self.slider_green.valueChanged.connect(self.get_ColorGREEN)
        self.slider_blue.valueChanged.connect(self.get_ColorBLUE)
        
        # Inisialisasi slider ke nilai 255
        self.slider_red.setValue(255)
        self.slider_green.setValue(255)
        self.slider_blue.setValue(255)
        
        self.color_red = 255
        self.color_green = 255
        self.color_blue = 255
        
        self.check_reset_button_state()
        
        print(self.color_red, self.color_green, self.color_blue)
        
        self.roi_x = 20
        self.roi_y = 20
        self.roi_w = 20
        self.roi_h = 20

    def check_reset_button_state(self):
        if self.slider_red.value() != 255 or self.slider_green.value() != 255 or self.slider_blue.value() != 255:
            self.btn_reset_rgb.setEnabled(True)
        else:
            self.btn_reset_rgb.setEnabled(False)

    def StartWebCam(self):
        try:
            self.textEdit.append(f"{self.DateTime.toString('d MMMM yy hh:mm:ss')}: Start Webcam ({self.camlist.currentText()})")
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)
            
            global camIndex
            camIndex = self.camlist.currentIndex()
            
            # Opencv QThread
            self.Worker1_Opencv = ThreadClass()
            self.Worker1_Opencv.ImageUpdate.connect(self.opencv_emit)
            self.Worker1_Opencv.FPS.connect(self.get_FPS)
            self.Worker1_Opencv.start()
            
        except Exception as error:
            print(str(error))
            self.textEdit.append(f"ERROR : {error}")
            
    def StopWebcam(self):
        self.textEdit.append(f"{self.DateTime.toString('d MMMM yy hh:mm:ss')}: Stop Webcam ({self.camlist.currentText()})")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.Worker1_Opencv.stop()
        
    def reset_rgb(self):
        self.slider_red.setValue(255)
        self.slider_green.setValue(255)
        self.slider_blue.setValue(255)

    @pyqtSlot(np.ndarray)
    def opencv_emit(self, Image):
        if self.slider_red.value() != 255 or self.slider_green.value() != 255 or self.slider_blue.value() != 255:
            modified_image = self.apply_rgb_overlay(Image)
        else:
            modified_image = Image
        original = self.cvt_cv_qt(modified_image)
        self.disp_main.setPixmap(original)
        self.disp_main.setScaledContents(True)
        
    def apply_rgb_overlay(self, image):
        overlay = np.zeros_like(image)
        overlay[:] = [self.color_blue, self.color_green, self.color_red]
        combined = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        return combined
    
    def get_ROIX(self, x):
        self.roi_x = x
    
    def get_ROIY(self, y):
        self.roi_y = y
    
    def get_ROIW(self, w):
        self.roi_w = w
    
    def get_ROIH(self, h):
        self.roi_h = h
        
    def get_ColorRED(self, red):
        self.color_red = red
        self.textEdit.append(f"Color Red Change Value : {self.color_red} -> {red}")
        self.check_reset_button_state()
        
    def get_ColorGREEN(self, green):
        self.color_green = green
        self.textEdit.append(f"Color Green Change Value : {self.color_green} -> {green}")
        self.check_reset_button_state()
        
    def get_ColorBLUE(self, blue):
        self.color_blue = blue
        self.textEdit.append(f"Color Blue Change Value : {self.color_blue} -> {blue}")
        self.check_reset_button_state()
   
    def cvt_cv_qt(self, Image):
        rgb_img = cv2.cvtColor(src=Image, code=cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        cvt2QtFormat = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(cvt2QtFormat)
    
    def getCPU_usage(self, cpu):
        self.Qlabel_cpu.setText(str(cpu) + " %")
        if cpu > 15: self.Qlabel_cpu.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25: self.Qlabel_cpu.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45: self.Qlabel_cpu.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65: self.Qlabel_cpu.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85: self.Qlabel_cpu.setStyleSheet("color: rgb(237, 85, 59);")

    def getRAM_usage(self, ram):
        self.Qlabel_ram.setText(str(ram[2]) + " %")
        if ram[2] > 15: self.Qlabel_ram.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25: self.Qlabel_ram.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45: self.Qlabel_ram.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65: self.Qlabel_ram.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85: self.Qlabel_ram.setStyleSheet("color: rgb(237, 85, 59);")

    def get_FPS(self, fps):
        self.Qlabel_fps.setText(str(fps))

    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock.display(self.DateTime.toString('hh:mm:ss'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
