from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDateTime, QTimer
from PyQt5 import QtGui
import os, cv2
import tensorflow
from keras.models import load_model
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


# klasifikasi model
print("============== Load Label Klasifikasi ==============")
LABEL_KLASIFIKASI = resource_path("label_klasifikasi.txt")
with open(LABEL_KLASIFIKASI, 'r') as f:
    labels_klasifikasi = [line.strip() for line in f.readlines()]
if labels_klasifikasi[0] == '???':
    del(labels_klasifikasi[0])
    
print("============== Load Model Klasifikasi ==============")
model_densenet = load_model(resource_path("model_densenet121_20eph.h5"))
print("| Load Model EfficientNet")
model_efficientnet = load_model(resource_path("model_efficientnet_20eph.h5"))
print("| Load Model Inception")
model_inception = load_model(resource_path("model_inception_20eph.h5"))
print("| Load Model Mobilenet")
model_mobilenet = load_model(resource_path("model_mobilenetv2_20eph.h5"))
print("| Load Model Resnet")
model_resnet = load_model(resource_path("model_resnet50_20eph.h5"))
print("| Load Model VGG")
model_vgg = load_model(resource_path("model_vgg16_20eph.h5"))
print("| Load Model Xception")
model_xception = load_model(resource_path("model_xception_20eph.h5"))

# END FIXX ========================================================================

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
        self.groupBox_14.setGeometry(QRect(10, 190, 581, 151))
        self.label_15 = QLabel(self.groupBox_14)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(20, 20, 551, 121))
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
        self.label_40.setPixmap(QPixmap(resource_path(u"klasifikasi_241.png")))
        self.label_40.setAlignment(Qt.AlignCenter)
        self.groupBox_17 = QGroupBox(self.frame_3)
        self.groupBox_17.setObjectName(u"groupBox_17")
        self.groupBox_17.setGeometry(QRect(10, 350, 551, 270))
        self.label_41 = QLabel(self.groupBox_17)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setGeometry(QRect(10, 20, 531, 241))
        self.label_41.setPixmap(QPixmap(resource_path(u"objek_deteksi_241.png")))
        self.label_41.setAlignment(Qt.AlignCenter)
        self.label_42 = QLabel(self.tab_3)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setGeometry(QRect(400, 0, 51, 51))
        self.label_42.setPixmap(QPixmap(resource_path(u"usd.png")))
        self.label_42.setScaledContents(True)
        self.label_43 = QLabel(self.tab_3)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setGeometry(QRect(460, 0, 51, 51))
        self.label_43.setPixmap(QPixmap(resource_path(u"logo51.png")))
        self.label_43.setScaledContents(True)
        self.label_37 = QLabel(self.tab_3)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setGeometry(QRect(20, 345, 561, 21))
        font6 = QFont()
        font6.setFamily(u"Nirmala UI")
        font6.setPointSize(10)
        font6.setBold(False)
        font6.setWeight(50)
        self.label_37.setFont(font6)
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
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.groupBox_6 = QGroupBox(self.tab_4)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setGeometry(QRect(10, 10, 461, 71))
        self.groupBox_6.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.label_18 = QLabel(self.groupBox_6)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(10, 10, 61, 51))
        self.label_18.setFont(font2)
        self.label_18.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.btn_resource_storage = QPushButton(self.groupBox_6)
        self.btn_resource_storage.setObjectName(u"btn_resource_storage")
        self.btn_resource_storage.setGeometry(QRect(230, 10, 100, 51))
        self.btn_resource_storage.setMinimumSize(QSize(100, 0))
        self.btn_resource_storage.setFont(font2)
        self.btn_resource_storage.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.btn_clear_klasifikasi = QPushButton(self.groupBox_6)
        self.btn_clear_klasifikasi.setObjectName(u"btn_clear_klasifikasi")
        self.btn_clear_klasifikasi.setGeometry(QRect(350, 10, 100, 51))
        self.btn_clear_klasifikasi.setMinimumSize(QSize(100, 0))
        self.btn_clear_klasifikasi.setFont(font2)
        self.btn_clear_klasifikasi.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.btn_start_3 = QPushButton(self.groupBox_6)
        self.btn_start_3.setObjectName(u"btn_start_3")
        self.btn_start_3.setGeometry(QRect(100, 10, 100, 51))
        self.btn_start_3.setMinimumSize(QSize(100, 0))
        self.btn_start_3.setFont(font2)
        self.btn_start_3.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.groupBox_7 = QGroupBox(self.tab_4)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setGeometry(QRect(10, 90, 241, 431))
        self.gridLayoutWidget_5 = QWidget(self.groupBox_7)
        self.gridLayoutWidget_5.setObjectName(u"gridLayoutWidget_5")
        self.gridLayoutWidget_5.setGeometry(QRect(20, 20, 211, 251))
        self.gridLayout_6 = QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.ckb_vgg = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_vgg.setObjectName(u"ckb_vgg")

        self.gridLayout_6.addWidget(self.ckb_vgg, 5, 0, 1, 1)

        self.ckb_densenet = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_densenet.setObjectName(u"ckb_densenet")
        self.ckb_densenet.setMinimumSize(QSize(0, 0))
        self.ckb_densenet.setSizeIncrement(QSize(0, 0))

        self.gridLayout_6.addWidget(self.ckb_densenet, 0, 0, 1, 1)

        self.ckb_restnet = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_restnet.setObjectName(u"ckb_restnet")

        self.gridLayout_6.addWidget(self.ckb_restnet, 4, 0, 1, 1)

        self.ckb_efficientnet = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_efficientnet.setObjectName(u"ckb_efficientnet")

        self.gridLayout_6.addWidget(self.ckb_efficientnet, 1, 0, 1, 1)

        self.ckb_inception = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_inception.setObjectName(u"ckb_inception")

        self.gridLayout_6.addWidget(self.ckb_inception, 2, 0, 1, 1)

        self.ckb_xception = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_xception.setObjectName(u"ckb_xception")

        self.gridLayout_6.addWidget(self.ckb_xception, 6, 0, 1, 1)

        self.ckb_mobilenet = QCheckBox(self.gridLayoutWidget_5)
        self.ckb_mobilenet.setObjectName(u"ckb_mobilenet")

        self.gridLayout_6.addWidget(self.ckb_mobilenet, 3, 0, 1, 1)

        self.btn_predict = QPushButton(self.groupBox_7)
        self.btn_predict.setObjectName(u"btn_predict")
        self.btn_predict.setGeometry(QRect(10, 330, 221, 25))
        font7 = QFont()
        font7.setFamily(u"Nirmala UI")
        font7.setPointSize(10)
        font7.setBold(True)
        font7.setWeight(75)
        self.btn_predict.setFont(font7)
        self.label_22 = QLabel(self.groupBox_7)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(10, 290, 151, 31))
        self.label_22.setFont(font7)
        self.ckb_show_rank = QCheckBox(self.groupBox_7)
        self.ckb_show_rank.setObjectName(u"ckb_show_rank")
        self.ckb_show_rank.setGeometry(QRect(160, 290, 70, 31))
        self.ckb_show_rank.setFont(font6)
        self.ckb_show_rank.setChecked(True)
        self.groupBox_8 = QGroupBox(self.tab_4)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setGeometry(QRect(260, 90, 421, 431))
        self.groupBox_8.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.disp_klasifikasi = QLabel(self.groupBox_8)
        self.disp_klasifikasi.setObjectName(u"disp_klasifikasi")
        self.disp_klasifikasi.setGeometry(QRect(10, 20, 400, 400))
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.disp_klasifikasi.sizePolicy().hasHeightForWidth())
        self.disp_klasifikasi.setSizePolicy(sizePolicy)
        self.disp_klasifikasi.setMinimumSize(QSize(400, 400))
        self.disp_klasifikasi.setStyleSheet(u"background-color: rgb(200, 247, 247);")
        self.disp_klasifikasi.setFrameShape(QFrame.StyledPanel)
        self.disp_klasifikasi.setFrameShadow(QFrame.Sunken)
        self.disp_klasifikasi.setAlignment(Qt.AlignCenter)
        self.groupBox_9 = QGroupBox(self.tab_4)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.groupBox_9.setGeometry(QRect(690, 10, 491, 511))
        self.gridLayoutWidget_8 = QWidget(self.groupBox_9)
        self.gridLayoutWidget_8.setObjectName(u"gridLayoutWidget_8")
        self.gridLayoutWidget_8.setGeometry(QRect(10, 340, 471, 161))
        self.gridLayout_9 = QGridLayout(self.gridLayoutWidget_8)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.result_rank_6 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_6.setObjectName(u"result_rank_6")
        self.result_rank_6.setFrameShape(QFrame.Box)
        self.result_rank_6.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_6, 1, 5, 1, 1)

        self.result_rank_1 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_1.setObjectName(u"result_rank_1")
        self.result_rank_1.setFrameShape(QFrame.Box)
        self.result_rank_1.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_1, 1, 0, 1, 1)

        self.rank_4 = QLabel(self.gridLayoutWidget_8)
        self.rank_4.setObjectName(u"rank_4")
        self.rank_4.setFrameShape(QFrame.Box)
        self.rank_4.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_4, 0, 3, 1, 1)

        self.rank_2 = QLabel(self.gridLayoutWidget_8)
        self.rank_2.setObjectName(u"rank_2")
        self.rank_2.setFrameShape(QFrame.Box)
        self.rank_2.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_2, 0, 1, 1, 1)

        self.rank_6 = QLabel(self.gridLayoutWidget_8)
        self.rank_6.setObjectName(u"rank_6")
        self.rank_6.setFrameShape(QFrame.Box)
        self.rank_6.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_6, 0, 5, 1, 1)

        self.rank_3 = QLabel(self.gridLayoutWidget_8)
        self.rank_3.setObjectName(u"rank_3")
        self.rank_3.setFrameShape(QFrame.Box)
        self.rank_3.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_3, 0, 2, 1, 1)

        self.result_rank_5 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_5.setObjectName(u"result_rank_5")
        self.result_rank_5.setFrameShape(QFrame.Box)
        self.result_rank_5.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_5, 1, 4, 1, 1)

        self.result_rank_2 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_2.setObjectName(u"result_rank_2")
        self.result_rank_2.setFrameShape(QFrame.Box)
        self.result_rank_2.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_2, 1, 1, 1, 1)

        self.rank_1 = QLabel(self.gridLayoutWidget_8)
        self.rank_1.setObjectName(u"rank_1")
        self.rank_1.setFrameShape(QFrame.Box)
        self.rank_1.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_1, 0, 0, 1, 1)

        self.result_rank_3 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_3.setObjectName(u"result_rank_3")
        self.result_rank_3.setFrameShape(QFrame.Box)
        self.result_rank_3.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_3, 1, 2, 1, 1)

        self.result_rank_4 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_4.setObjectName(u"result_rank_4")
        self.result_rank_4.setFrameShape(QFrame.Box)
        self.result_rank_4.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_4, 1, 3, 1, 1)

        self.rank_5 = QLabel(self.gridLayoutWidget_8)
        self.rank_5.setObjectName(u"rank_5")
        self.rank_5.setMinimumSize(QSize(0, 0))
        self.rank_5.setFrameShape(QFrame.Box)
        self.rank_5.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_5, 0, 4, 1, 1)

        self.rank_7 = QLabel(self.gridLayoutWidget_8)
        self.rank_7.setObjectName(u"rank_7")
        self.rank_7.setFrameShape(QFrame.Box)
        self.rank_7.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.rank_7, 0, 6, 1, 1)

        self.result_rank_7 = QLabel(self.gridLayoutWidget_8)
        self.result_rank_7.setObjectName(u"result_rank_7")
        self.result_rank_7.setFrameShape(QFrame.Box)
        self.result_rank_7.setAlignment(Qt.AlignCenter)

        self.gridLayout_9.addWidget(self.result_rank_7, 1, 6, 1, 1)

        self.groupBox_11 = QGroupBox(self.groupBox_9)
        self.groupBox_11.setObjectName(u"groupBox_11")
        self.groupBox_11.setGeometry(QRect(10, 100, 471, 231))
        self.label_34 = QLabel(self.groupBox_11)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(10, 10, 451, 16))
        font8 = QFont()
        font8.setFamily(u"Nirmala UI")
        font8.setPointSize(12)
        font8.setBold(True)
        font8.setWeight(75)
        self.label_34.setFont(font8)
        self.label_34.setAlignment(Qt.AlignCenter)
        self.top_predict = QLabel(self.groupBox_11)
        self.top_predict.setObjectName(u"top_predict")
        self.top_predict.setGeometry(QRect(10, 40, 451, 181))
        self.top_predict.setFont(font7)
        self.top_predict.setFrameShape(QFrame.Box)
        self.top_predict.setAlignment(Qt.AlignCenter)
        self.groupBox_12 = QGroupBox(self.groupBox_9)
        self.groupBox_12.setObjectName(u"groupBox_12")
        self.groupBox_12.setGeometry(QRect(10, 19, 471, 71))
        self.label_35 = QLabel(self.groupBox_12)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(10, 10, 451, 16))
        self.label_35.setFont(font8)
        self.label_35.setAlignment(Qt.AlignCenter)
        self.filename_input = QLabel(self.groupBox_12)
        self.filename_input.setObjectName(u"filename_input")
        self.filename_input.setGeometry(QRect(10, 30, 451, 31))
        self.filename_input.setFont(font7)
        self.filename_input.setFrameShape(QFrame.Box)
        self.filename_input.setAlignment(Qt.AlignCenter)
        self.lcd_clock_3 = QLCDNumber(self.tab_4)
        self.lcd_clock_3.setObjectName(u"lcd_clock_3")
        self.lcd_clock_3.setGeometry(QRect(480, 10, 201, 71))
        self.lcd_clock_3.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.lcd_clock_3.setDigitCount(8)
        self.groupBox_10 = QGroupBox(self.tab_4)
        self.groupBox_10.setObjectName(u"groupBox_10")
        self.groupBox_10.setGeometry(QRect(10, 520, 1171, 131))
        self.groupBox_10.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.textEdit_klasifikasi = QTextEdit(self.groupBox_10)
        self.textEdit_klasifikasi.setObjectName(u"textEdit_klasifikasi")
        self.textEdit_klasifikasi.setGeometry(QRect(10, 20, 1151, 101))
        self.textEdit_klasifikasi.setStyleSheet(u"background-color: rgb(255, 247, 247);\n"
"border-radius:3px;")
        self.tabWidget.addTab(self.tab_4, "")
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
        self.label_38.setPixmap(QPixmap(resource_path(u"me.png")))
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
        self.label_20.setPixmap(QPixmap(resource_path(u"laravel.png")))
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
        self.label_21.setPixmap(QPixmap(resource_path(u"git.png")))
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
        self.label_24.setPixmap(QPixmap(resource_path(u"mongodb.png")))
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
        self.label_25.setPixmap(QPixmap(resource_path(u"sql.png")))
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
        self.label_27.setPixmap(QPixmap(resource_path(u"nodejs.png")))
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
        self.label_28.setPixmap(QPixmap(resource_path(u"terminal.png")))
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
        self.label_19.setPixmap(QPixmap(resource_path(u"python.png")))
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
        self.label_23.setPixmap(QPixmap(resource_path(u"java.png")))
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
        self.label_32.setPixmap(QPixmap(resource_path(u"qt.png")))
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
        self.label_33.setPixmap(QPixmap(resource_path(u"cursor.svg")))
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
        
        # ===================================================================================
        # Bagian TAB 1 | Home
        # ===================================================================================
        Object_Detection.setWindowIcon(QtGui.QIcon(resource_path('logo.png')))
        Object_Detection.resize(1193, 696)
        Object_Detection.setStyleSheet(u"background-color: rgb(237, 250, 239);")
        Object_Detection.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # Setup QTimer
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start()
        
        # ===================================================================================
        # Bagian TAB 2 | Klasification
        # ===================================================================================
        # masalah file
        self.file_ready = ""
        # self.btn_resource_storage.clicked.connect(self.resource_storage)
        self.btn_resource_storage.clicked.connect(lambda: self.resource_storage(Object_Detection))
        self.btn_clear_klasifikasi.setEnabled(False)
        self.btn_clear_klasifikasi.clicked.connect(self.clear_klasifikasi)
        self.btn_predict.clicked.connect(self.klasifikasi_aksara)
        self.btn_predict.setEnabled(False)  # Initially disable the predict button
        self.ckb_show_rank.stateChanged.connect(self.show_rank)
        self.clear_rank_result()
        
        # Connect checkbox state changes to enable/disable predict button
        self.ckb_densenet.setEnabled(False)
        self.ckb_efficientnet.setEnabled(False)
        self.ckb_inception.setEnabled(False)
        self.ckb_mobilenet.setEnabled(False)
        self.ckb_restnet.setEnabled(False)
        self.ckb_vgg.setEnabled(False)
        self.ckb_xception.setEnabled(False)
        
        self.ckb_densenet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_efficientnet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_inception.stateChanged.connect(self.update_predict_button_state)
        self.ckb_mobilenet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_restnet.stateChanged.connect(self.update_predict_button_state)
        self.ckb_vgg.stateChanged.connect(self.update_predict_button_state)
        self.ckb_xception.stateChanged.connect(self.update_predict_button_state)
        
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 
        
        self.model_densenet = None
        self.model_efficientnet = None
        self.model_inception = None
        self.model_mobilenet = None
        self.model_restnet = None
        self.model_vgg = None
        self.model_xception = None

        self.retranslateUi(Object_Detection)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Object_Detection)
    # setupUi
    
    # ====================================================================================
    # Bagian Setting All
    # ====================================================================================
    
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
    
    def resource_storage(self,MainWindow):
        fname = QFileDialog.getOpenFileName(MainWindow, 'Open File', 'C:', 'Image Files (*.png *.jpg *.jpeg)')
        if fname[0]:
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: File loaded: {fname[0]}")
            self.file_ready = fname[0]
            self.show_image()
            self.ckb_densenet.setEnabled(True)
            self.ckb_efficientnet.setEnabled(True)
            self.ckb_inception.setEnabled(True)
            self.ckb_mobilenet.setEnabled(True)
            self.ckb_restnet.setEnabled(True)
            self.ckb_vgg.setEnabled(True)
            self.ckb_xception.setEnabled(True)
            self.btn_clear_klasifikasi.setEnabled(True)
        else:
            self.textEdit_klasifikasi.append(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: No file selected")
        
    def show_image(self):
        self.filename_input.setText(self.file_ready)
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
                        title.setVisible(True)
                        label.setVisible(True)
                        title.setText(f"Rank {i+1}")
                        label.setText(f"{arr_pred_sorted[i][2]} \n {arr_pred_sorted[i][0]} \n {arr_pred_sorted[i][1]}")
                    else:
                        title.setVisible(False)
                        label.setVisible(False)
                        
        elif len(arr_pred) == 1:
            rank_title = [self.rank_1, self.rank_2, self.rank_3, self.rank_4, self.rank_5, self.rank_6, self.rank_7]
            rank_result = [self.result_rank_1, self.result_rank_2, self.result_rank_3, self.result_rank_4, self.result_rank_5, self.result_rank_6, self.result_rank_7]
            for i, (title, label) in enumerate(zip(rank_title, rank_result)):
                title.setVisible(False)
                label.setVisible(False)
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
        
    def clear_klasifikasi(self):
        self.filename_input.setText("")
        self.clear_rank_result()
        self.ckb_densenet.setEnabled(False)
        self.ckb_efficientnet.setEnabled(False)
        self.ckb_inception.setEnabled(False)
        self.ckb_mobilenet.setEnabled(False)
        self.ckb_restnet.setEnabled(False)
        self.ckb_vgg.setEnabled(False)
        self.ckb_xception.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.file_ready= ""
        self.btn_clear_klasifikasi.setEnabled(False)
        
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
        
    def getRAM_usage(self,ram):
        self.Qlabel_ram.setText(str(ram[2]) + " %")
        if ram[2] > 15: self.Qlabel_ram.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25: self.Qlabel_ram.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45: self.Qlabel_ram.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65: self.Qlabel_ram.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85: self.Qlabel_ram.setStyleSheet("color: rgb(237, 85, 59);")
    
    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock_2.display(self.DateTime.toString('hh:mm:ss'))
        self.lcd_clock_3.display(self.DateTime.toString('hh:mm:ss'))

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
        self.groupBox_15.setTitle(QCoreApplication.translate("Object_Detection", u"JawaLearn Application Features ", None))
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
        self.label_37.setText(QCoreApplication.translate("Object_Detection", u"This version of the application only provides Javanese script image classification features", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Object_Detection", u"Home", None))
        self.groupBox_6.setTitle("")
        self.label_18.setText(QCoreApplication.translate("Object_Detection", u"Resource", None))
        self.btn_resource_storage.setText(QCoreApplication.translate("Object_Detection", u"STORAGE", None))
        self.btn_clear_klasifikasi.setText(QCoreApplication.translate("Object_Detection", u"CLEAR", None))
        self.btn_start_3.setText(QCoreApplication.translate("Object_Detection", u"CAMERA", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("Object_Detection", u"Select Model To Use Predict", None))
        self.ckb_vgg.setText(QCoreApplication.translate("Object_Detection", u"VGG16", None))
        self.ckb_densenet.setText(QCoreApplication.translate("Object_Detection", u"DenseNet121", None))
        self.ckb_restnet.setText(QCoreApplication.translate("Object_Detection", u"ResNet50", None))
        self.ckb_efficientnet.setText(QCoreApplication.translate("Object_Detection", u"EfficientNet", None))
        self.ckb_inception.setText(QCoreApplication.translate("Object_Detection", u"Inception", None))
        self.ckb_xception.setText(QCoreApplication.translate("Object_Detection", u"Xception", None))
        self.ckb_mobilenet.setText(QCoreApplication.translate("Object_Detection", u"MobileNetV2", None))
        self.btn_predict.setText(QCoreApplication.translate("Object_Detection", u"PREDICT", None))
        self.label_22.setText(QCoreApplication.translate("Object_Detection", u"SHOW RANK PREDICT :", None))
        self.ckb_show_rank.setText(QCoreApplication.translate("Object_Detection", u"Active", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("Object_Detection", u"MAIN WINDOW", None))
        self.disp_klasifikasi.setText("")
        self.groupBox_9.setTitle(QCoreApplication.translate("Object_Detection", u"TOP 5 PREDICTED", None))
        self.result_rank_6.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.result_rank_1.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.rank_4.setText(QCoreApplication.translate("Object_Detection", u"RANK 4", None))
        self.rank_2.setText(QCoreApplication.translate("Object_Detection", u"RANK 2", None))
        self.rank_6.setText(QCoreApplication.translate("Object_Detection", u"RANK 6", None))
        self.rank_3.setText(QCoreApplication.translate("Object_Detection", u"RANK 3", None))
        self.result_rank_5.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.result_rank_2.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.rank_1.setText(QCoreApplication.translate("Object_Detection", u"RANK 1", None))
        self.result_rank_3.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.result_rank_4.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.rank_5.setText(QCoreApplication.translate("Object_Detection", u"RANK 5", None))
        self.rank_7.setText(QCoreApplication.translate("Object_Detection", u"RANK 7", None))
        self.result_rank_7.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.groupBox_11.setTitle("")
        self.label_34.setText(QCoreApplication.translate("Object_Detection", u"TOP PREDICT", None))
        self.top_predict.setText(QCoreApplication.translate("Object_Detection", u"result", None))
        self.groupBox_12.setTitle("")
        self.label_35.setText(QCoreApplication.translate("Object_Detection", u"File Name ", None))
        self.filename_input.setText("")
        self.groupBox_10.setTitle(QCoreApplication.translate("Object_Detection", u"LOG", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("Object_Detection", u"Klasifikasi Aksara", None))
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
    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # ui = Ui_Object_Detection()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec_())
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    main_window.center()
    sys.exit(app.exec_())