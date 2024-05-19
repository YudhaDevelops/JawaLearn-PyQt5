from PySide2 import QtWidgets
from ui_Main_Window import Ui_Object_Detection

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_Object_Detection()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


