from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, pyqtSlot, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import numpy as np
import psutil
import sys
import sysinfo
import tensorflow as tf
import time
import importlib.util
from threading import Thread

# FIX GAK BOLEH DI GANGGU GUGAT ===================================================

# Kelas VideoStream untuk menangani streaming video dari webcam
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30, camIndex=0):
        self.stream = cv2.VideoCapture(camIndex)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Parameter input langsung di dalam kode
MODEL_PATH = "./models/detectObject/model.tflite"
LABEL_PATH = "./models/detectObject/labels.txt"
min_conf_threshold = 0.5
resW, resH = '640', '480'
imW, imH = int(resW), int(resH)
use_TPU = False

# Mengimpor pustaka TensorFlow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Load the label map
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()
# END FIXX ========================================================================

class boardInfoClass(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)
    temp = pyqtSignal(float)
    
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            #temp = sysinfo.getTemp()
            self.cpu.emit(cpu)
            self.ram.emit(ram)
            #self.temp.emit(temp)

    def stop(self):
        self.ThreadActive = False
        self.quit()
        
#   QLabel display
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("Main_Window.ui",self)
        self.textEdit.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Load model")
        self.textEdit.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Model loaded")
        
        self.online_cam = QCameraInfo.availableCameras()
        self.camlist.addItems([c.description() for c in self.online_cam])
        
        self.btn_start.clicked.connect(self.StartWebCam)
        self.btn_stop.clicked.connect(self.StopWebcam)
        self.btn_reset_rgb.setEnabled(False)
        self.btn_reset_rgb.clicked.connect(self.reset_rgb)
        
        self.resource_usage = boardInfoClass()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.getCPU_usage)
        self.resource_usage.ram.connect(self.getRAM_usage)
        
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 
        
        # Atribut untuk menyimpan nilai RGB
        self.color_red = 255
        self.color_green = 255
        self.color_blue = 255

        self.slider_red.setValue(255)
        self.slider_green.setValue(255)
        self.slider_blue.setValue(255)
        
        self.slider_red.valueChanged.connect(self.set_red)
        self.slider_green.valueChanged.connect(self.set_green)
        self.slider_blue.valueChanged.connect(self.set_blue)
    
    def StartWebCam(self):
        try:
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)
            
            global kameraIndeks
            kameraIndeks = self.camlist.currentIndex()
            self.videostream = VideoStream(resolution=(imW, imH), framerate=30, camIndex=kameraIndeks).start()
            self.textEdit.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Start Webcam ({self.camlist.currentText()})")
            self.detect_and_display()

        except Exception as error :
            pass
    def StopWebcam(self):
        currentDateTime = QDateTime.currentDateTime()
        self.textEdit.append(f"{currentDateTime.toString('d MMMM yy hh:mm:ss')}: Stop Webcam ({self.camlist.currentText()})")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if hasattr(self, 'videostream'):
            self.videostream.stop()
        self.timer.stop()  # Menghentikan QTimer yang memicu deteksi dan tampilan
        
        
    def detect_and_display(self):
        global frame_rate_calc
        t1 = cv2.getTickCount()
        frame1 = self.videostream.read()
        frame = frame1.copy()
        
        # Sesuaikan frame dengan nilai RGB dari slider
        frame_rgb = self.adjust_rgb(frame)
        
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                
                if self.ckb_show_predict.isChecked():
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                if self.ckb_show_predict.isChecked():
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # fps
        self.set_FPS(frame_rate_calc)
        
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        self.disp_main.setPixmap(pixmap)
        self.disp_main.setScaledContents(True)
        self.disp_main.setFixedSize(imW, imH)

        # QTimer.singleShot(10, self.detect_and_display)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_and_display)
        self.timer.start(10)
        
    def adjust_rgb(self, frame):
        # Ubah nilai RGB frame sesuai dengan nilai slider
        if self.color_red != 255:
            frame[:, :, 2] = np.clip(frame[:, :, 2] * (self.color_red / 255.0), 0, 255)  # Red
        if self.color_green != 255:
            frame[:, :, 1] = np.clip(frame[:, :, 1] * (self.color_green / 255.0), 0, 255)  # Green
        if self.color_blue != 255:
            frame[:, :, 0] = np.clip(frame[:, :, 0] * (self.color_blue / 255.0), 0, 255)  # Blue
        return frame
    
    def reset_rgb(self):
        if self.color_red != 255:
            self.slider_red.setValue(255)
        if self.color_green != 255:
            self.slider_green.setValue(255)
        if self.color_blue != 255:
            self.slider_blue.setValue(255)
        
    def set_red(self):
        self.color_red = self.slider_red.value()
        self.check_rgb_sliders()
    
    def set_green(self):
        self.color_green = self.slider_green.value()
        self.check_rgb_sliders()
        
    def set_blue(self):
        self.color_blue = self.slider_blue.value()
        self.check_rgb_sliders()
        
    def check_rgb_sliders(self):
        if self.color_red != 255 or self.color_green != 255 or self.color_blue != 255:
            self.btn_reset_rgb.setEnabled(True)
        else:
            self.btn_reset_rgb.setEnabled(False)
        
    def set_FPS(self,fps):
        self.Qlabel_fps.setText('{0:.2f}'.format(fps))
        if fps > 5: self.Qlabel_fps.setStyleSheet("color: rgb(237, 85, 59);")
        if fps > 15: self.Qlabel_fps.setStyleSheet("color: rgb(60, 174, 155);")
        if fps > 25: self.Qlabel_fps.setStyleSheet("color: rgb(85, 170, 255);")
        if fps > 35: self.Qlabel_fps.setStyleSheet("color: rgb(23, 63, 95);")
    
    def getCPU_usage(self,cpu):
        self.Qlabel_cpu.setText(str(cpu) + " %")
        if cpu > 15: self.Qlabel_cpu.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25: self.Qlabel_cpu.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45: self.Qlabel_cpu.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65: self.Qlabel_cpu.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85: self.Qlabel_cpu.setStyleSheet("color: rgb(237, 85, 59);")

    def getRAM_usage(self,ram):
        self.Qlabel_ram.setText(str(ram[2]) + " %")
        if ram[2] > 15: self.Qlabel_ram.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25: self.Qlabel_ram.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45: self.Qlabel_ram.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65: self.Qlabel_ram.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85: self.Qlabel_ram.setStyleSheet("color: rgb(237, 85, 59);")
    
    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock.display(self.DateTime.toString('hh:mm:ss'))
        self.lcd_clock_2.display(self.DateTime.toString('hh:mm:ss'))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Apakah Anda yakin ingin menutup aplikasi?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            if hasattr(self, 'videostream'):
                self.videostream.stop()
            print('Window closed')
        else:
            event.ignore()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
# conda activate conda39 && E: && cd E:\LEGENA\_build_app\JawaLearn-PyQt5 && python main.py
