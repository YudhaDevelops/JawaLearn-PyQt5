from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, pyqtSlot, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import numpy as np
import psutil
import sys
import sysinfo
from PIL import Image, ImageOps
import tensorflow as tf
from keras.models import load_model
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

# klasifikasi model
print("============== Load Label Klasifikasi ==============")
LABEL_KLASIFIKASI = "./models/modelA/class_names_legena_2.txt"
with open(LABEL_KLASIFIKASI, 'r') as f:
    labels_klasifikasi = [line.strip() for line in f.readlines()]
if labels_klasifikasi[0] == '???':
    del(labels_klasifikasi[0])
    
print("============== Load Model Klasifikasi ==============")
model_densenet = load_model("./models/modelA/model_densenet121_20eph.h5")
model_efficientnet = load_model("./models/modelA/model_efficientnet_20eph.h5")
model_inception = load_model("./models/modelA/model_inception_20eph.h5")
model_mobilenet = load_model("./models/modelA/model_mobilenetv2_20eph.h5")
model_resnet = load_model("./models/modelA/model_resnet50_20eph.h5")
model_vgg = load_model("./models/modelA/model_vgg16_20eph.h5")
model_xception = load_model("./models/modelA/model_xception_20eph.h5")


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

print("============== Load Label Object Detection ==============")
# Load the label map
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])
    
print("============== Load Model Object Detection ==============")
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
        # ===================================================================================
        # Bagian TAB 2 | Klasification
        # ===================================================================================
        # masalah file
        self.file_ready = ""
        self.btn_resource_storage.clicked.connect(self.resource_storage)
        self.btn_predict.clicked.connect(self.klasifikasi_aksara)
        self.btn_predict.setEnabled(False)  # Initially disable the predict button
        self.ckb_show_rank.stateChanged.connect(self.show_rank)
        self.clear_rank_result()
        
        # Connect checkbox state changes to enable/disable predict button
        self.ckb_densenet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_efficientnet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_inception.stateChanged.connect(self.update_predict_button_state)
        self.ckb_mobilenet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_restnet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_vgg.stateChanged.connect(self.update_predict_button_state)
        self.ckb_xception.stateChanged.connect(self.update_predict_button_state)
        
        # ===================================================================================
        # Bagian TAB 3 | Object Detection
        # ===================================================================================
        
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
        
        self.model_densenet = None
        self.model_efficientnet = None
        self.model_inception = None
        self.model_mobilenet = None
        self.model_restnet = None
        self.model_vgg = None
        self.model_xception = None
        
        self.slider_red.valueChanged.connect(self.set_red)
        self.slider_green.valueChanged.connect(self.set_green)
        self.slider_blue.valueChanged.connect(self.set_blue)
    
    # ====================================================================================
    # Bagian TAB 2 | Klasification
    # ====================================================================================
    def update_predict_button_state(self):
        # Enable the predict button if any model checkbox is checked
        self.btn_predict.setEnabled(
            self.ckb_densenet.isChecked() or
            self.ckb_efficientnet.isChecked() or
            self.ckb_inception.isChecked() or
            self.ckb_mobilenet.isChecked() or
            self.ckb_restnet.isChecked() or
            self.ckb_vgg.isChecked() or
            self.ckb_xception.isChecked()
        )
    
    def resource_storage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File','E:', 'Image Files (*.png *.jpg *.jpeg)')
        self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: File loaded: {fname[0]}")
        self.file_ready = fname[0]
        self.show_image()
        
    def show_image(self):
        self.img = cv2.imread(self.file_ready)
        self.img = cv2.resize(self.img, (400,400))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(self.img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        self.disp_klasifikasi.setPixmap(pixmap)
        self.disp_klasifikasi.setScaledContents(True)
        self.disp_klasifikasi.setFixedSize(400, 400)
        
    def klasifikasi_aksara(self):
        arr_pred = []
        if self.ckb_densenet.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model DenseNet121 | Load model success")
            model = model_densenet
            predict_class, skor = self.klasifikasi(model)
            self.model_densenet = model
            arr_pred.append([predict_class, skor,"DenseNet121"])
            
        if self.ckb_efficientnet.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model Efficientnet | Load model success")
            model = model_efficientnet
            predict_class, skor = self.klasifikasi(model)
            self.model_efficientnet = model
            arr_pred.append([predict_class, skor,"EfficientNet"])
            
        if self.ckb_inception.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model Inception | Load model success")
            model = model_inception
            predict_class, skor = self.klasifikasi(model)
            self.model_inception = model
            arr_pred.append([predict_class, skor,"Inception"])
            
        if self.ckb_mobilenet.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model MobileNetV2 | Load model success")
            model = model_mobilenet
            predict_class, skor = self.klasifikasi(model)
            self.model_mobilenet = model
            arr_pred.append([predict_class, skor,"MobileNetV2"])
            
        if self.ckb_restnet.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model RestNet50 | Load model success")
            model = model_resnet
            predict_class, skor = self.klasifikasi(model)
            self.model_restnet = model
            arr_pred.append([predict_class, skor,"RestNet50"])

        if self.ckb_vgg.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model VGG16 | Load model success")
            model = model_vgg
            predict_class, skor = self.klasifikasi(model)
            self.model_vgg = model
            arr_pred.append([predict_class, skor,"VGG16"])

        if self.ckb_xception.isChecked():
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict from model Xception | Load model success")
            model = model_xception
            predict_class, skor = self.klasifikasi(model)
            self.model_xception = model
            arr_pred.append([predict_class, skor,"Xception"])

        
        if len(arr_pred) > 1:
            max_value = max(arr_pred, key=lambda x: float(x[1]))
            top_label = max_value[0]
            top_nilai = max_value[1]
            top_model = max_value[2]
            print(top_label,top_nilai,top_model)
            self.top_predict.setText(f"Model Name : {top_model} \n Label : {top_label} \n Confidence : {top_nilai}")
            
            arr_pred_sorted = sorted(arr_pred, key=lambda x: float(x[1]), reverse=True)
            
            if self.ckb_show_rank.isChecked():
                rank_title = [self.rank_1, self.rank_2, self.rank_3, self.rank_4, self.rank_5, self.rank_6, self.rank_7]
                rank_result = [self.result_rank_1, self.result_rank_2, self.result_rank_3, self.result_rank_4, self.result_rank_5, self.result_rank_6, self.result_rank_7]
                for i, (title, label) in enumerate(zip(rank_title, rank_result)):
                    
                    if i < len(arr_pred_sorted):
                        title.setText(f"Rank {i+1}")
                        label.setText(f"{arr_pred_sorted[i][2]} \n {arr_pred_sorted[i][0]} \n {arr_pred_sorted[i][1]}")
                    else:
                        title.setVisible(False)
                        label.setVisible(False)
                        
        elif len(arr_pred) == 1:
            label_terbesar = arr_pred[0][0]
            nilai_terbesar = arr_pred[0][1]
            model_name = arr_pred[0][2]
            self.top_predict.setText(f"Model Name : {model_name} \n Label : {label_terbesar} \n Confidence : {nilai_terbesar}")
            
        arr_pred.clear()
        
    def clear_rank_result(self):
        self.top_predict.setText("")
        rank_result = [self.result_rank_1, self.result_rank_2, self.result_rank_3, self.result_rank_4, self.result_rank_5, self.result_rank_6, self.result_rank_7]
        for label in rank_result:
            label.setText("")
        
    def show_rank(self):
        pass
        # if self.ckb_show_rank.isChecked():
            # self.result_rank_1.setText(f"{max_value[i][2]} \n {max_value[i][0]} \n {max_value[i][1]}")
            
    
    def klasifikasi(self,model):
        size = (224, 224)
        self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Proses predict | Read image resource")
        image_loaded = Image.open(self.file_ready).convert("RGB")
        image = ImageOps.fit(image_loaded, size, Image.Resampling.LANCZOS)

        # convert image to array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # set model input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load the image into the array
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        predict_class = labels_klasifikasi[index]
        confidence_score = prediction[0][index]
        skor_dua_angka = "{:.2f}".format(confidence_score)
        return predict_class,skor_dua_angka
        
        
    
    # ====================================================================================
    # Bagian TAB 3 | Object Detection
    # ====================================================================================
    
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
        self.lcd_clock_3.display(self.DateTime.toString('hh:mm:ss'))

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
