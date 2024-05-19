from PySide2.QtCore import QTimer
import sys
import cv2
import numpy as np
import tensorflow as tf
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./models/detectObject/model.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class Ui_Object_Detection(object):
    def setupUi(self, Object_Detection):
        if not Object_Detection.objectName():
            Object_Detection.setObjectName(u"Object_Detection")
        Object_Detection.resize(1186, 634)
        Object_Detection.setStyleSheet(u"background-color: rgb(237, 250, 239);")
        Object_Detection.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.centralwidget = QWidget(Object_Detection)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 1191, 981))
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.groupBox_5 = QGroupBox(self.tab_5)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(10, 90, 241, 501))
        self.gridLayoutWidget_3 = QWidget(self.groupBox_5)
        self.gridLayoutWidget_3.setObjectName(u"gridLayoutWidget_3")
        self.gridLayoutWidget_3.setGeometry(QRect(10, 240, 221, 80))
        self.gridLayout_3 = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.label_9 = QLabel(self.gridLayoutWidget_3)
        self.label_9.setObjectName(u"label_9")
        font = QFont()
        font.setFamily(u"Nirmala UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)

        self.gridLayout_3.addWidget(self.label_9, 2, 0, 1, 1)

        self.check_show_predict = QCheckBox(self.gridLayoutWidget_3)
        self.check_show_predict.setObjectName(u"check_show_predict")
        self.check_show_predict.setChecked(True)

        self.gridLayout_3.addWidget(self.check_show_predict, 2, 1, 1, 1)

        self.label_8 = QLabel(self.gridLayoutWidget_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)

        self.gridLayout_3.addWidget(self.label_8, 0, 0, 1, 1)

        self.checkBox = QCheckBox(self.gridLayoutWidget_3)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setChecked(True)

        self.gridLayout_3.addWidget(self.checkBox, 0, 1, 1, 1)

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
        self.value_green.setFont(font)
        self.value_green.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.value_green, 3, 2, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_4, 4, 0, 1, 1)

        self.label_4 = QLabel(self.gridLayoutWidget_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

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
        self.value_red.setFont(font)
        self.value_red.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.value_red, 1, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(3, 3, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.value_blue = QLabel(self.gridLayoutWidget_2)
        self.value_blue.setObjectName(u"value_blue")
        self.value_blue.setMinimumSize(QSize(15, 0))
        self.value_blue.setFont(font)
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
        self.label_6.setFont(font)

        self.gridLayout_2.addWidget(self.label_6, 3, 0, 1, 1)

        self.label_7 = QLabel(self.gridLayoutWidget_2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)

        self.gridLayout_2.addWidget(self.label_7, 5, 0, 1, 1)

        self.btn_reset_rgb = QPushButton(self.gridLayoutWidget_2)
        self.btn_reset_rgb.setObjectName(u"btn_reset_rgb")
        self.btn_reset_rgb.setFont(font)

        self.gridLayout_2.addWidget(self.btn_reset_rgb, 7, 0, 1, 3)

        self.groupBox_3 = QGroupBox(self.tab_5)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(260, 80, 661, 511))
        self.groupBox_3.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.disp_main = QLabel(self.groupBox_3)
        self.disp_main.setObjectName(u"disp_main")
        self.disp_main.setGeometry(QRect(10, 20, 640, 480))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.disp_main.sizePolicy().hasHeightForWidth())
        self.disp_main.setSizePolicy(sizePolicy)
        self.disp_main.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.disp_main.setAlignment(Qt.AlignCenter)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tabWidget.addTab(self.tab_2, "")

        Object_Detection.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Object_Detection)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1186, 26))
        Object_Detection.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Object_Detection)
        self.statusbar.setObjectName(u"statusbar")
        Object_Detection.setStatusBar(self.statusbar)

        self.retranslateUi(Object_Detection)
        self.tabWidget.setCurrentIndex(4)

        QMetaObject.connectSlotsByName(Object_Detection)

    def retranslateUi(self, Object_Detection):
        Object_Detection.setWindowTitle(QCoreApplication.translate("Object_Detection", u"Object Detection", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Object_Detection", u"Tab 1", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("Object_Detection", u"Tab 2", None))
        self.label_9.setText(QCoreApplication.translate("Object_Detection", u"Prediction", None))
        self.check_show_predict.setText("")
        self.label_8.setText(QCoreApplication.translate("Object_Detection", u"Color mask", None))
        self.checkBox.setText("")
        self.value_green.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.label_4.setText(QCoreApplication.translate("Object_Detection", u"Red", None))
        self.value_red.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.value_blue.setText(QCoreApplication.translate("Object_Detection", u"0", None))
        self.label_6.setText(QCoreApplication.translate("Object_Detection", u"Green", None))
        self.label_7.setText(QCoreApplication.translate("Object_Detection", u"Blue", None))
        self.btn_reset_rgb.setText(QCoreApplication.translate("Object_Detection", u"Reset", None))
        self.disp_main.setText(QCoreApplication.translate("Object_Detection", u"TextLabel", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), QCoreApplication.translate("Object_Detection", u"Home", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Object_Detection", u"Tab 3", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Object_Detection", u"Tab 4", None))

class MainWindow(QMainWindow, Ui_Object_Detection):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect signals and slots
        self.slider_red.valueChanged.connect(self.update_values)
        self.slider_green.valueChanged.connect(self.update_values)
        self.slider_blue.valueChanged.connect(self.update_values)
        self.btn_reset_rgb.clicked.connect(self.reset_values)

        # Initialize video capture
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Unable to open video source")
            sys.exit()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_values(self):
        self.value_red.setText(str(self.slider_red.value()))
        self.value_green.setText(str(self.slider_green.value()))
        self.value_blue.setText(str(self.slider_blue.value()))

    def reset_values(self):
        self.slider_red.setValue(0)
        self.slider_green.setValue(0)
        self.slider_blue.setValue(0)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Error: Unable to read frame")
            return

        rgb_values = {
            'r': self.slider_red.value(),
            'g': self.slider_green.value(),
            'b': self.slider_blue.value()
        }

        frame_with_objects = self.object_detection(frame, rgb_values)

        image = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.disp_main.setPixmap(QPixmap.fromImage(qImg))

    def object_detection(self, frame, rgb_values):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(frame_rgb, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if scores[i] > 0.5:  # Threshold for object detection
                ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
                xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
                ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
                xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (rgb_values['b'], rgb_values['g'], rgb_values['r']), 2)
                label = f"{int(class_ids[i])}: {int(scores[i]*100)}%"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (rgb_values['b'], rgb_values['g'], rgb_values['r']), 2)

        return frame

    def closeEvent(self, event):
        self.capture.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
