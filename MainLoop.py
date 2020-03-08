import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui_Widget import Ui_Form
import cv2
import numpy as np
from ui_DialogMainLoop import Ui_MainWindow
from Widget import *
from captureThreading import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Face_Recognition")

        self.myWidget = VideoDisplayWidget(
            "haarcascade_frontalface_default.xml")

        self.captureThread = QThread()
        self.captureWorker = RecordVideo()
        self.captureWorker.moveToThread(self.captureThread)
        self.captureWorker.imageData.connect(self.myWidget.image_data_slot)
        self.captureThread.started.connect(self.captureWorker.startRecord)
        self.captureThread.start()

        self.ui.verticalLayout.addWidget(self.myWidget)


if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    main = MainWindow()
    main.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
