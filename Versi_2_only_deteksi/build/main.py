import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDateTime, QTimer
from PyQt5 import QtGui
import importlib.util
import psutil, os, cv2
import tensorflow as tf
from keras.models import load_model
from threading import Thread
import numpy as np
import sys
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Parameter input langsung di dalam kode
MODEL_PATH = resource_path("model_objek_deteksi.tflite")
LABEL_PATH = resource_path("label_objek_deteksi.txt")
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Object_Detection()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False) 
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'JawaLearn', 'Apakah Anda yakin ingin menutup aplikasi?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            if hasattr(self.ui, 'videostream'):
                self.ui.videostream.stop()
            print('Window closed')
        else:
            event.ignore()
            
    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
        
class Ui_Object_Detection(object):
    def setupUi(self, Object_Detection):
        if not Object_Detection.objectName():
            Object_Detection.setObjectName(u"Object_Detection")
        
        self.centralwidget = QWidget(Object_Detection)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 1191, 981))
        self.tabWidget.setStyleSheet(u"")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.label_9 = QLabel(self.tab_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(10, 10, 341, 31))
        font = QFont()
        font.setFamily(u"Nirmala UI")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.groupBox_13 = QGroupBox(self.tab_3)
        self.groupBox_13.setObjectName(u"groupBox_13")
        self.groupBox_13.setGeometry(QRect(10, 50, 581, 121))
        self.label_36 = QLabel(self.groupBox_13)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setGeometry(QRect(10, 20, 561, 91))
        font1 = QFont()
        font1.setFamily(u"Times New Roman")
        font1.setPointSize(10)
        self.label_36.setFont(font1)
        self.label_36.setLineWidth(1)
        self.label_36.setTextFormat(Qt.PlainText)
        self.label_36.setScaledContents(False)
        self.label_36.setWordWrap(True)
        self.groupBox_14 = QGroupBox(self.tab_3)
        self.groupBox_14.setObjectName(u"groupBox_14")
        self.groupBox_14.setGeometry(QRect(10, 190, 581, 141))
        self.label_15 = QLabel(self.groupBox_14)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(20, 20, 551, 111))
        self.label_15.setFont(font1)
        self.label_15.setWordWrap(True)
        self.groupBox_15 = QGroupBox(self.tab_3)
        self.groupBox_15.setObjectName(u"groupBox_15")
        self.groupBox_15.setGeometry(QRect(10, 370, 581, 271))
        self.frame_15 = QFrame(self.groupBox_15)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setGeometry(QRect(290, 30, 281, 231))
        font2 = QFont()
        font2.setPointSize(10)
        self.frame_15.setFont(font2)
        self.frame_15.setStyleSheet(u"")
        self.frame_15.setFrameShape(QFrame.Box)
        self.frame_15.setFrameShadow(QFrame.Raised)
        self.label_26 = QLabel(self.frame_15)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(10, 10, 261, 31))
        font3 = QFont()
        font3.setFamily(u"Segoe UI Black")
        font3.setPointSize(12)
        font3.setBold(True)
        font3.setWeight(75)
        self.label_26.setFont(font3)
        self.label_26.setAlignment(Qt.AlignCenter)
        self.label_30 = QLabel(self.frame_15)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(20, 70, 251, 131))
        font4 = QFont()
        font4.setFamily(u"Times New Roman")
        font4.setPointSize(11)
        self.label_30.setFont(font4)
        self.label_30.setWordWrap(True)
        self.frame_7 = QFrame(self.groupBox_15)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setGeometry(QRect(10, 30, 281, 231))
        self.frame_7.setStyleSheet(u"")
        self.frame_7.setFrameShape(QFrame.Box)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.label_16 = QLabel(self.frame_7)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(10, 10, 261, 31))
        self.label_16.setFont(font3)
        self.label_16.setAlignment(Qt.AlignCenter)
        self.label_29 = QLabel(self.frame_7)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(20, 70, 241, 131))
        self.label_29.setFont(font4)
        self.label_29.setWordWrap(True)
        self.frame_3 = QFrame(self.tab_3)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setGeometry(QRect(610, 10, 571, 631))
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.label_39 = QLabel(self.frame_3)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setGeometry(QRect(10, 10, 551, 51))
        font5 = QFont()
        font5.setFamily(u"Times New Roman")
        font5.setPointSize(11)
        font5.setBold(True)
        font5.setWeight(75)
        self.label_39.setFont(font5)
        self.label_39.setAlignment(Qt.AlignCenter)
        self.label_39.setWordWrap(True)
        self.groupBox_16 = QGroupBox(self.frame_3)
        self.groupBox_16.setObjectName(u"groupBox_16")
        self.groupBox_16.setGeometry(QRect(10, 80, 551, 270))
        self.label_40 = QLabel(self.groupBox_16)
        self.label_40.setObjectName(u"label_40")
        self.label_40.setGeometry(QRect(10, 20, 531, 241))
        self.label_40.setPixmap(QPixmap(resource_path("klasifikasi_241.png")))
        self.label_40.setAlignment(Qt.AlignCenter)
        self.groupBox_17 = QGroupBox(self.frame_3)
        self.groupBox_17.setObjectName(u"groupBox_17")
        self.groupBox_17.setGeometry(QRect(10, 350, 551, 270))
        self.label_41 = QLabel(self.groupBox_17)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setGeometry(QRect(10, 20, 531, 241))
        self.label_41.setPixmap(QPixmap(resource_path("objek_deteksi_241.png")))
        self.label_41.setAlignment(Qt.AlignCenter)
        self.label_42 = QLabel(self.tab_3)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setGeometry(QRect(400, 0, 51, 51))
        self.label_42.setPixmap(QPixmap(resource_path("usd.png")))
        self.label_42.setScaledContents(True)
        self.label_43 = QLabel(self.tab_3)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setGeometry(QRect(460, 0, 51, 51))
        self.label_43.setPixmap(QPixmap(resource_path("logo51.png")))
        self.label_43.setScaledContents(True)
        self.label_37 = QLabel(self.tab_3)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setGeometry(QRect(10, 340, 581, 21))
        self.label_37.setStyleSheet(u"background-color: rgb(235, 235, 0);")
        self.label_37.setAlignment(Qt.AlignCenter)
        self.tabWidget.addTab(self.tab_3, "")
        self.groupBox_15.raise_()
        self.label_9.raise_()
        self.groupBox_13.raise_()
        self.groupBox_14.raise_()
        self.frame_3.raise_()
        self.label_42.raise_()
        self.label_43.raise_()
        self.label_37.raise_()
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.groupBox_5 = QGroupBox(self.tab_5)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(10, 90, 241, 501))
        self.gridLayoutWidget_3 = QWidget(self.groupBox_5)
        self.gridLayoutWidget_3.setObjectName(u"gridLayoutWidget_3")
        self.gridLayoutWidget_3.setGeometry(QRect(10, 260, 221, 31))
        self.gridLayout_3 = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_8 = QLabel(self.gridLayoutWidget_3)
        self.label_8.setObjectName(u"label_8")
        font6 = QFont()
        font6.setFamily(u"Nirmala UI")
        font6.setPointSize(10)
        font6.setBold(True)
        font6.setWeight(75)
        self.label_8.setFont(font6)

        self.gridLayout_3.addWidget(self.label_8, 0, 0, 1, 1)

        self.ckb_show_predict = QCheckBox(self.gridLayoutWidget_3)
        self.ckb_show_predict.setObjectName(u"ckb_show_predict")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ckb_show_predict.sizePolicy().hasHeightForWidth())
        self.ckb_show_predict.setSizePolicy(sizePolicy)
        self.ckb_show_predict.setMinimumSize(QSize(0, 0))
        self.ckb_show_predict.setSizeIncrement(QSize(0, 0))
        self.ckb_show_predict.setBaseSize(QSize(0, 0))
        font7 = QFont()
        font7.setPointSize(12)
        font7.setBold(False)
        font7.setWeight(50)
        self.ckb_show_predict.setFont(font7)
        self.ckb_show_predict.setLayoutDirection(Qt.LeftToRight)
        self.ckb_show_predict.setChecked(True)

        self.gridLayout_3.addWidget(self.ckb_show_predict, 0, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.gridLayoutWidget_2 = QWidget(self.groupBox_5)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(10, 20, 223, 211))
        self.gridLayout_2 = QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_5, 6, 0, 1, 1)

        self.value_green = QLabel(self.gridLayoutWidget_2)
        self.value_green.setObjectName(u"value_green")
        self.value_green.setMinimumSize(QSize(15, 0))
        self.value_green.setFont(font6)
        self.value_green.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.value_green, 3, 2, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_4, 4, 0, 1, 1)

        self.label_4 = QLabel(self.gridLayoutWidget_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font6)

        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 2, 0, 1, 1)

        self.slider_green = QSlider(self.gridLayoutWidget_2)
        self.slider_green.setObjectName(u"slider_green")
        self.slider_green.setMaximum(255)
        self.slider_green.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.slider_green, 3, 1, 1, 1)

        self.value_red = QLabel(self.gridLayoutWidget_2)
        self.value_red.setObjectName(u"value_red")
        self.value_red.setMinimumSize(QSize(37, 0))
        self.value_red.setFont(font6)
        self.value_red.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.value_red, 1, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(3, 3, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.value_blue = QLabel(self.gridLayoutWidget_2)
        self.value_blue.setObjectName(u"value_blue")
        self.value_blue.setMinimumSize(QSize(15, 0))
        self.value_blue.setFont(font6)
        self.value_blue.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.value_blue, 5, 2, 1, 1)

        self.slider_red = QSlider(self.gridLayoutWidget_2)
        self.slider_red.setObjectName(u"slider_red")
        self.slider_red.setMaximum(255)
        self.slider_red.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.slider_red, 1, 1, 1, 1)

        self.slider_blue = QSlider(self.gridLayoutWidget_2)
        self.slider_blue.setObjectName(u"slider_blue")
        self.slider_blue.setMaximum(255)
        self.slider_blue.setSliderPosition(0)
        self.slider_blue.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.slider_blue, 5, 1, 1, 1)

        self.label_6 = QLabel(self.gridLayoutWidget_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font6)

        self.gridLayout_2.addWidget(self.label_6, 3, 0, 1, 1)

        self.label_7 = QLabel(self.gridLayoutWidget_2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font6)

        self.gridLayout_2.addWidget(self.label_7, 5, 0, 1, 1)

        self.btn_reset_rgb = QPushButton(self.gridLayoutWidget_2)
        self.btn_reset_rgb.setObjectName(u"btn_reset_rgb")
        self.btn_reset_rgb.setFont(font6)

        self.gridLayout_2.addWidget(self.btn_reset_rgb, 7, 0, 1, 3)

        self.groupBox_3 = QGroupBox(self.tab_5)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(260, 80, 661, 511))
        self.groupBox_3.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.disp_main = QLabel(self.groupBox_3)
        self.disp_main.setObjectName(u"disp_main")
        self.disp_main.setGeometry(QRect(10, 20, 640, 480))
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.disp_main.sizePolicy().hasHeightForWidth())
        self.disp_main.setSizePolicy(sizePolicy1)
        self.disp_main.setMinimumSize(QSize(640, 480))
        self.disp_main.setStyleSheet(u"background-color: rgb(200, 247, 247);")
        self.disp_main.setFrameShape(QFrame.StyledPanel)
        self.disp_main.setFrameShadow(QFrame.Sunken)
        self.disp_main.setAlignment(Qt.AlignCenter)
        self.groupBox_4 = QGroupBox(self.tab_5)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(930, 0, 251, 591))
        self.groupBox_4.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.textEdit = QTextEdit(self.groupBox_4)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(10, 20, 231, 561))
        self.textEdit.setStyleSheet(u"background-color: rgb(246, 247, 247);\n"
"border-radius:3px;")
        self.lcd_clock = QLCDNumber(self.tab_5)
        self.lcd_clock.setObjectName(u"lcd_clock")
        self.lcd_clock.setGeometry(QRect(720, 10, 201, 71))
        self.lcd_clock.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.lcd_clock.setDigitCount(8)
        self.groupBox_2 = QGroupBox(self.tab_5)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(450, 10, 261, 71))
        self.groupBox_2.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.gridLayoutWidget = QWidget(self.groupBox_2)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(10, 10, 241, 51))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.Qlabel_cpu = QLabel(self.gridLayoutWidget)
        self.Qlabel_cpu.setObjectName(u"Qlabel_cpu")
        font8 = QFont()
        font8.setPointSize(10)
        font8.setBold(True)
        font8.setWeight(75)
        self.Qlabel_cpu.setFont(font8)
        self.Qlabel_cpu.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_cpu, 1, 1, 1, 1)

        self.label_3 = QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font8)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_2 = QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font8)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.Qlabel_fps = QLabel(self.gridLayoutWidget)
        self.Qlabel_fps.setObjectName(u"Qlabel_fps")
        self.Qlabel_fps.setFont(font8)
        self.Qlabel_fps.setStyleSheet(u"color: rgb(237, 85, 59);")
        self.Qlabel_fps.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_fps, 1, 0, 1, 1)

        self.label_5 = QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font8)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)

        self.Qlabel_ram = QLabel(self.gridLayoutWidget)
        self.Qlabel_ram.setObjectName(u"Qlabel_ram")
        self.Qlabel_ram.setFont(font8)
        self.Qlabel_ram.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_ram, 1, 2, 1, 1)

        self.groupBox = QGroupBox(self.tab_5)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 431, 71))
        self.groupBox.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 10, 51, 51))
        self.label.setFont(font2)
        self.label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.camlist = QComboBox(self.groupBox)
        self.camlist.setObjectName(u"camlist")
        self.camlist.setGeometry(QRect(70, 10, 133, 51))
        self.camlist.setMinimumSize(QSize(133, 10))
        self.camlist.setMaximumSize(QSize(100, 16777215))
        self.camlist.setFont(font2)
        self.camlist.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.btn_start = QPushButton(self.groupBox)
        self.btn_start.setObjectName(u"btn_start")
        self.btn_start.setGeometry(QRect(210, 10, 100, 51))
        self.btn_start.setMinimumSize(QSize(100, 0))
        self.btn_start.setFont(font2)
        self.btn_start.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.btn_stop = QPushButton(self.groupBox)
        self.btn_stop.setObjectName(u"btn_stop")
        self.btn_stop.setGeometry(QRect(320, 10, 100, 51))
        self.btn_stop.setMinimumSize(QSize(100, 0))
        self.btn_stop.setFont(font2)
        self.btn_stop.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QWidget()
        self.tab_6.setObjectName(u"tab_6")
        self.frame = QFrame(self.tab_6)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(0, 0, 371, 591))
        self.frame.setStyleSheet(u"")
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Raised)
        self.label_10 = QLabel(self.frame)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(10, 340, 351, 21))
        font9 = QFont()
        font9.setFamily(u"Nirmala UI")
        font9.setPointSize(16)
        font9.setBold(True)
        font9.setWeight(75)
        self.label_10.setFont(font9)
        self.label_10.setStyleSheet(u"")
        self.label_10.setAlignment(Qt.AlignCenter)
        self.label_11 = QLabel(self.frame)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 360, 351, 31))
        font10 = QFont()
        font10.setFamily(u"Nirmala UI")
        font10.setPointSize(11)
        font10.setBold(False)
        font10.setWeight(50)
        self.label_11.setFont(font10)
        self.label_11.setStyleSheet(u"")
        self.label_11.setAlignment(Qt.AlignCenter)
        self.frame_16 = QFrame(self.frame)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setGeometry(QRect(80, 110, 220, 220))
        font11 = QFont()
        font11.setPointSize(7)
        self.frame_16.setFont(font11)
        self.frame_16.setStyleSheet(u"border-radius:100px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_16.setFrameShape(QFrame.Box)
        self.frame_16.setFrameShadow(QFrame.Sunken)
        self.frame_16.setLineWidth(3)
        self.label_38 = QLabel(self.frame_16)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setGeometry(QRect(10, 10, 200, 200))
        self.label_38.setPixmap(QPixmap(resource_path("me.png")))
        self.label_38.setScaledContents(True)
        self.frame_2 = QFrame(self.tab_6)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setGeometry(QRect(370, 0, 811, 591))
        self.frame_2.setFrameShape(QFrame.Box)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.label_12 = QLabel(self.frame_2)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(50, 90, 431, 31))
        font12 = QFont()
        font12.setFamily(u"Times New Roman")
        font12.setPointSize(17)
        self.label_12.setFont(font12)
        self.label_13 = QLabel(self.frame_2)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(50, 120, 571, 31))
        self.label_13.setFont(font12)
        self.label_14 = QLabel(self.frame_2)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(50, 160, 731, 151))
        self.label_14.setFont(font12)
        self.label_14.setWordWrap(True)
        self.label_17 = QLabel(self.frame_2)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(50, 440, 751, 31))
        self.label_17.setFont(font12)
        self.frame_5 = QFrame(self.frame_2)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setGeometry(QRect(120, 480, 51, 51))
        self.frame_5.setFont(font11)
        self.frame_5.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_5.setFrameShape(QFrame.Box)
        self.frame_5.setFrameShadow(QFrame.Sunken)
        self.frame_5.setLineWidth(3)
        self.label_20 = QLabel(self.frame_5)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(10, 10, 31, 31))
        self.label_20.setPixmap(QPixmap(resource_path("laravel.png")))
        self.label_20.setScaledContents(True)
        self.frame_6 = QFrame(self.frame_2)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setGeometry(QRect(190, 480, 51, 51))
        self.frame_6.setFont(font11)
        self.frame_6.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_6.setFrameShape(QFrame.Box)
        self.frame_6.setFrameShadow(QFrame.Sunken)
        self.frame_6.setLineWidth(3)
        self.label_21 = QLabel(self.frame_6)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(10, 10, 31, 31))
        self.label_21.setPixmap(QPixmap(resource_path("git.png")))
        self.label_21.setScaledContents(True)
        self.frame_9 = QFrame(self.frame_2)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setGeometry(QRect(330, 480, 51, 51))
        self.frame_9.setFont(font11)
        self.frame_9.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_9.setFrameShape(QFrame.Box)
        self.frame_9.setFrameShadow(QFrame.Sunken)
        self.frame_9.setLineWidth(3)
        self.label_24 = QLabel(self.frame_9)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(10, 10, 31, 31))
        self.label_24.setPixmap(QPixmap(resource_path("mongodb.png")))
        self.label_24.setScaledContents(True)
        self.frame_10 = QFrame(self.frame_2)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setGeometry(QRect(400, 480, 51, 51))
        self.frame_10.setFont(font11)
        self.frame_10.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_10.setFrameShape(QFrame.Box)
        self.frame_10.setFrameShadow(QFrame.Sunken)
        self.frame_10.setLineWidth(3)
        self.label_25 = QLabel(self.frame_10)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(10, 10, 31, 31))
        self.label_25.setPixmap(QPixmap(resource_path("sql.png")))
        self.label_25.setScaledContents(True)
        self.frame_11 = QFrame(self.frame_2)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setGeometry(QRect(470, 480, 51, 51))
        self.frame_11.setFont(font11)
        self.frame_11.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_11.setFrameShape(QFrame.Box)
        self.frame_11.setFrameShadow(QFrame.Sunken)
        self.frame_11.setLineWidth(3)
        self.label_27 = QLabel(self.frame_11)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(10, 10, 31, 31))
        self.label_27.setPixmap(QPixmap(resource_path("nodejs.png")))
        self.label_27.setScaledContents(True)
        self.frame_12 = QFrame(self.frame_2)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setGeometry(QRect(540, 480, 51, 51))
        self.frame_12.setFont(font11)
        self.frame_12.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_12.setFrameShape(QFrame.Box)
        self.frame_12.setFrameShadow(QFrame.Sunken)
        self.frame_12.setLineWidth(3)
        self.label_28 = QLabel(self.frame_12)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(10, 10, 31, 31))
        self.label_28.setPixmap(QPixmap(resource_path("terminal.png")))
        self.label_28.setScaledContents(True)
        self.frame_4 = QFrame(self.frame_2)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setGeometry(QRect(50, 480, 51, 51))
        self.frame_4.setFont(font11)
        self.frame_4.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_4.setFrameShape(QFrame.Box)
        self.frame_4.setFrameShadow(QFrame.Sunken)
        self.frame_4.setLineWidth(3)
        self.label_19 = QLabel(self.frame_4)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(10, 10, 31, 31))
        self.label_19.setPixmap(QPixmap(resource_path("python.png")))
        self.label_19.setScaledContents(True)
        self.frame_8 = QFrame(self.frame_2)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setGeometry(QRect(260, 480, 51, 51))
        self.frame_8.setFont(font11)
        self.frame_8.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_8.setFrameShape(QFrame.Box)
        self.frame_8.setFrameShadow(QFrame.Sunken)
        self.frame_8.setLineWidth(3)
        self.label_23 = QLabel(self.frame_8)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(10, 10, 31, 31))
        self.label_23.setPixmap(QPixmap(resource_path("java.png")))
        self.label_23.setScaledContents(True)
        self.label_31 = QLabel(self.frame_2)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(50, 320, 741, 91))
        self.label_31.setFont(font12)
        self.label_31.setWordWrap(True)
        self.frame_13 = QFrame(self.frame_2)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setGeometry(QRect(610, 480, 51, 51))
        self.frame_13.setFont(font11)
        self.frame_13.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_13.setFrameShape(QFrame.Box)
        self.frame_13.setFrameShadow(QFrame.Sunken)
        self.frame_13.setLineWidth(3)
        self.label_32 = QLabel(self.frame_13)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(10, 10, 31, 31))
        self.label_32.setPixmap(QPixmap(resource_path("qt.png")))
        self.label_32.setScaledContents(True)
        self.frame_14 = QFrame(self.frame_2)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setGeometry(QRect(680, 480, 51, 51))
        self.frame_14.setFont(font11)
        self.frame_14.setStyleSheet(u"border-radius:20px;\n"
"background-color: rgb(194, 194, 194);")
        self.frame_14.setFrameShape(QFrame.Box)
        self.frame_14.setFrameShadow(QFrame.Sunken)
        self.frame_14.setLineWidth(3)
        self.label_33 = QLabel(self.frame_14)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(10, 10, 31, 31))
        self.label_33.setPixmap(QPixmap(resource_path("cursor.svg")))
        self.label_33.setScaledContents(True)
        self.lcd_clock_2 = QLCDNumber(self.frame_2)
        self.lcd_clock_2.setObjectName(u"lcd_clock_2")
        self.lcd_clock_2.setGeometry(QRect(610, 0, 201, 71))
        self.lcd_clock_2.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.lcd_clock_2.setDigitCount(8)
        self.tabWidget.addTab(self.tab_6, "")
        Object_Detection.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(Object_Detection)
        self.statusbar.setObjectName(u"statusbar")
        Object_Detection.setStatusBar(self.statusbar)

        self.retranslateUi(Object_Detection)
        self.slider_red.valueChanged.connect(self.value_red.setNum)
        self.slider_blue.valueChanged.connect(self.value_blue.setNum)
        self.slider_green.valueChanged.connect(self.value_green.setNum)
        # ===================================================================================
        # Bagian TAB 1 | Home
        # ===================================================================================
        Object_Detection.setWindowIcon(QtGui.QIcon(resource_path('logo.png')))
        Object_Detection.resize(1193, 696)
        Object_Detection.setStyleSheet(u"background-color: rgb(237, 250, 239);")
        Object_Detection.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # ram cpu
        self.resource_usage = boardInfoClass()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.getCPU_usage)
        self.resource_usage.ram.connect(self.getRAM_usage)
        
        # Setup QTimer
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start()
        
        
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
        

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Object_Detection)
    # setupUi
    
    # ====================================================================================
    # Bagian Setting All
    # ====================================================================================
        
    
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

    def retranslateUi(self, Object_Detection):
        Object_Detection.setWindowTitle(QCoreApplication.translate("Object_Detection", u"Object Detection", None))
        self.label_9.setText(QCoreApplication.translate("Object_Detection", u"Welcome to the JawaLearn Application", None))
        self.groupBox_13.setTitle(QCoreApplication.translate("Object_Detection", u"Overall understanding", None))
        self.label_36.setText(QCoreApplication.translate("Object_Detection", u"The JawaLearn application is an application built using machine learning, namely CNN (Convolutional Neural Network), and using image data as a learning tool for the machine. By training different machines it is hoped that it will help you understand the form of Javanese script.", None))
        self.groupBox_14.setTitle(QCoreApplication.translate("Object_Detection", u"Purpose of Building the Application", None))
        self.label_15.setText(QCoreApplication.translate("Object_Detection", u"1. Makes it easy for you to learn Javanese script \n"
"2. Creating an Augmented Reality application by combining AI technology which can make teaching and learning Javanese script easier. \n"
"3. Proving that Augmented Reality technology can be combined together using the Convolutional \n"
"    Neural Network (CNN) algorithm.\n"
"4. Contribute to overcoming students' difficulties in learning Javanese script using Augmented Reality \n"
"    technology", None))
        self.groupBox_15.setTitle(QCoreApplication.translate("Object_Detection", u"JawaLearn Application Features", None))
        self.label_26.setText(QCoreApplication.translate("Object_Detection", u"AR Detection of Javanese Script", None))
        self.label_30.setText(QCoreApplication.translate("Object_Detection", u"This feature can be used to transliterate Javanese LEGENA characters with real-time detection using the available camera. This feature gives meaning to the characters placed in front of the camera.", None))
        self.label_16.setText(QCoreApplication.translate("Object_Detection", u"Classification of Javanese Script", None))
        self.label_29.setText(QCoreApplication.translate("Object_Detection", u"This feature can be used to transliterate Javanese LEGENA characters with detection based on images of written characters entered either on the camera daci or the storage on your device.", None))
        self.label_39.setText(QCoreApplication.translate("Object_Detection", u"APPLICATION OF AUGMENTED REALITY TECHNOLOGY USING CNN (CONVOLUTIONAL NEURAL NETWORK) IN LEARNING JAVANESE CHARACTERS", None))
        self.groupBox_16.setTitle(QCoreApplication.translate("Object_Detection", u"Classification of Javanese Script", None))
        self.label_40.setText("")
        self.groupBox_17.setTitle(QCoreApplication.translate("Object_Detection", u"AR Detection of Javanese Script", None))
        self.label_41.setText("")
        self.label_42.setText("")
        self.label_43.setText("")
        self.label_37.setText(QCoreApplication.translate("Object_Detection", u"This version of the application only provides Javanese script image AR Detection features", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Object_Detection", u"Home", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Object_Detection", u"Setting", None))
        self.label_8.setText(QCoreApplication.translate("Object_Detection", u"SHOW PREDICTION :", None))
        self.ckb_show_predict.setText(QCoreApplication.translate("Object_Detection", u"Active", None))
        self.value_green.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.label_4.setText(QCoreApplication.translate("Object_Detection", u"RED  :", None))
        self.value_red.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.value_blue.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.label_6.setText(QCoreApplication.translate("Object_Detection", u"GREEN :", None))
        self.label_7.setText(QCoreApplication.translate("Object_Detection", u"BLUE :", None))
        self.btn_reset_rgb.setText(QCoreApplication.translate("Object_Detection", u"RESET", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Object_Detection", u"MAIN WINDOW", None))
        self.disp_main.setText("")
        self.groupBox_4.setTitle(QCoreApplication.translate("Object_Detection", u"LOG", None))
        self.groupBox_2.setTitle("")
        self.Qlabel_cpu.setText(QCoreApplication.translate("Object_Detection", u"0.0", None))
        self.label_3.setText(QCoreApplication.translate("Object_Detection", u"CPU", None))
        self.label_2.setText(QCoreApplication.translate("Object_Detection", u"FPS", None))
        self.Qlabel_fps.setText(QCoreApplication.translate("Object_Detection", u"0.0", None))
        self.label_5.setText(QCoreApplication.translate("Object_Detection", u"RAM", None))
        self.Qlabel_ram.setText(QCoreApplication.translate("Object_Detection", u"0.0", None))
        self.groupBox.setTitle("")
        self.label.setText(QCoreApplication.translate("Object_Detection", u"Camera", None))
        self.btn_start.setText(QCoreApplication.translate("Object_Detection", u"START", None))
        self.btn_stop.setText(QCoreApplication.translate("Object_Detection", u"STOP", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), QCoreApplication.translate("Object_Detection", u"AR Jawa Learn", None))
        self.label_10.setText(QCoreApplication.translate("Object_Detection", u"FX. BIMA YUDHA PRATAMA", None))
        self.label_11.setText(QCoreApplication.translate("Object_Detection", u"Informatika Angkaatan 20", None))
        self.label_38.setText("")
        self.label_12.setText(QCoreApplication.translate("Object_Detection", u"I'm a student from Sanata Dharma University", None))
        self.label_13.setText(QCoreApplication.translate("Object_Detection", u"My major is informatics with a focus on intelligent machines", None))
        self.label_14.setText(QCoreApplication.translate("Object_Detection", u"I'm easily adapt to different topic about Back-End Web Developer,Small Front-End Developer , Entry-level designer , Desktop applications depending on what the project requires. I love exploring something new about technology  and use them to build a\u00a0cool\u00a0stuffs", None))
        self.label_17.setText(QCoreApplication.translate("Object_Detection", u" Languages and Tools:", None))
        self.label_20.setText("")
        self.label_21.setText("")
        self.label_24.setText("")
        self.label_25.setText("")
        self.label_27.setText("")
        self.label_28.setText("")
        self.label_19.setText("")
        self.label_23.setText("")
        self.label_31.setText(QCoreApplication.translate("Object_Detection", u"<html><head/><body><p align=\"justify\">I already tried to deploy the JawaLearn application on the Streamlit website, but the AR can't be run because there are problems with the package used.</p><p align=\"justify\">You can see it at <a href=\"https://jawalearn.streamlit.app\"><span style=\" text-decoration: underline; color:#0000ff;\">JawaLearn.streamlit.app</span></a></p></body></html>", None))
        self.label_32.setText("")
        self.label_33.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), QCoreApplication.translate("Object_Detection", u"About Me", None))
    # retranslateUi

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    main_window.center()
    sys.exit(app.exec_())