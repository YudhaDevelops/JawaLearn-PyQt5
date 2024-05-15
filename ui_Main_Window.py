from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


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
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.disp_main.sizePolicy().hasHeightForWidth())
        self.disp_main.setSizePolicy(sizePolicy)
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
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(True)
        font1.setWeight(75)
        self.Qlabel_cpu.setFont(font1)
        self.Qlabel_cpu.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_cpu, 1, 1, 1, 1)

        self.label_3 = QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_2 = QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font1)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.Qlabel_fps = QLabel(self.gridLayoutWidget)
        self.Qlabel_fps.setObjectName(u"Qlabel_fps")
        self.Qlabel_fps.setFont(font1)
        self.Qlabel_fps.setStyleSheet(u"color: rgb(237, 85, 59);")
        self.Qlabel_fps.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_fps, 1, 0, 1, 1)

        self.label_5 = QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)

        self.Qlabel_ram = QLabel(self.gridLayoutWidget)
        self.Qlabel_ram.setObjectName(u"Qlabel_ram")
        self.Qlabel_ram.setFont(font1)
        self.Qlabel_ram.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.Qlabel_ram, 1, 2, 1, 1)

        self.groupBox = QGroupBox(self.tab_5)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 431, 71))
        self.groupBox.setStyleSheet(u"background-color: rgb(246, 247, 247);")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 10, 51, 51))
        font2 = QFont()
        font2.setPointSize(10)
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
        self.tabWidget.addTab(self.tab_6, "")
        Object_Detection.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(Object_Detection)
        self.statusbar.setObjectName(u"statusbar")
        Object_Detection.setStatusBar(self.statusbar)

        self.retranslateUi(Object_Detection)
        self.slider_red.valueChanged.connect(self.value_red.setNum)
        self.slider_blue.valueChanged.connect(self.value_blue.setNum)
        self.slider_green.valueChanged.connect(self.value_green.setNum)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(Object_Detection)
    # setupUi

    def retranslateUi(self, Object_Detection):
        Object_Detection.setWindowTitle(QCoreApplication.translate("Object_Detection", u"Object Detection", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Object_Detection", u"Home", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("Object_Detection", u"Klasifikasi Aksara", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Object_Detection", u"Setting", None))
        self.label_9.setText(QCoreApplication.translate("Object_Detection", u"Show Location Predict :", None))
        self.check_show_predict.setText(QCoreApplication.translate("Object_Detection", u"Show", None))
        self.label_8.setText(QCoreApplication.translate("Object_Detection", u"SHOW LABEL :", None))
        self.checkBox.setText(QCoreApplication.translate("Object_Detection", u"Show", None))
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
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), QCoreApplication.translate("Object_Detection", u"About Me", None))
    # retranslateUi

