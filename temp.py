from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import psutil
import sys
import tensorflow as tf
import time

class MainLogic(QThread):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, camera_index):
        super(MainLogic, self).__init__()
        self.camera_index = camera_index
        self.PATH_TO_MODEL = "./models/detectObject/model.tflite"
        self.PATH_TO_LABELS = "./models/detectObject/labels.txt"
        self.labels = self.load_labels()
        self.interpreter = self.load_tf_lite_model()
        self.min_conf_threshold = 0.5

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.input_mean = 127.5
        self.input_std = 127.5
        self.outname = self.output_details[0]['name']
        if 'StatefulPartitionedCall' in self.outname:  # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

    def load_tf_lite_model(self):
        try:
            print("Start Process load model...")
            interpreter = tf.lite.Interpreter(model_path=self.PATH_TO_MODEL)
            interpreter.allocate_tensors()
            print("Finish process load model...")
            return interpreter
        except ValueError as ve:
            print("Error loading the TensorFlow Lite model:", ve)
            exit()

    def load_labels(self):
        print("Start Process load labels...")
        with open(self.PATH_TO_LABELS, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print("Finish process load labels...")
        return labels

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        prev_frame_time = 0
        new_frame_time = 0
        while True:
            ret, frame_cap = self.cap.read()
            if ret:
                new_frame_time = time.time()
                fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time
                self.image_data.emit(frame_cap)
                time.sleep(0.01)  # Delay untuk mengurangi beban CPU
            else:
                print("Frame not available")

    def predict(self, frame):
        resW, resH = 640, 480
        imW, imH = int(resW), int(resH)
        frame_resized = cv2.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]  # Confidence of detected objects
        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                # print(label)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text
        return frame


class ThreadClass(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FPS = pyqtSignal(int)

    def __init__(self, camera_index):
        super(ThreadClass, self).__init__()
        self.camera_index = camera_index

    def run(self):
        self.main_logic = MainLogic(self.camera_index)
        self.main_logic.start()
        self.main_logic.image_data.connect(self.ImageUpdate.emit)
        self.main_logic.finished.connect(self.quit)
        self.main_logic.finished.connect(self.main_logic.deleteLater)
        self.main_logic.finished.connect(self.main_logic.cap.release)

    def stop(self):
        self.main_logic.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("Main_Window.ui", self)
        self.online_cam = QCameraInfo.availableCameras()
        self.camlist.addItems([c.description() for c in self.online_cam])
        self.btn_start.clicked.connect(self.StartWebCam)
        self.btn_stop.clicked.connect(self.StopWebcam)
        self.btn_stop.setEnabled(False)

        self.resource_usage = ThreadClass()
        self.resource_usage.FPS.connect(self.get_FPS)
        self.resource_usage.ImageUpdate.connect(self.display_image)
        self.resource_usage.start()

    def StartWebCam(self):
        try:
            print(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Start Webcam ({self.camlist.currentText()})")
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)
            self.resource_usage.camera_index = self.camlist.currentIndex()
            self.resource_usage.start()
        except Exception as error:
            print("ERROR :", error)

    def StopWebcam(self):
        print(f"{QDateTime.currentDateTime().toString('d MMMM yy hh:mm:ss')}: Stop Webcam ({self.camlist.currentText()})")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.resource_usage.stop()

    def display_image(self, frame):
        if self.check_show_predict.isChecked():
            frame = self.resource_usage.main_logic.predict(frame)
        self.show_frame(frame)

    def show_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.disp_main.setPixmap(pixmap)
        self.disp_main.setScaledContents(True)

    def get_FPS(self, fps):
        self.Qlabel_fps.setText(str(fps))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
