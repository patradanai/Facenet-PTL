import sys
import os
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

    # directory of folder
    dir = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Face_Recognition")

        # self.Mywidget = VideoDisplayWidget(
        #     "haarcascade_frontalface_default.xml")

        self.captureThread = QThread()
        self.captureWorker = RecordVideo()
        self.captureWorker.moveToThread(self.captureThread)
        self.captureWorker.imageData.connect(self.ui.widget.image_data_slot)
        self.captureThread.started.connect(self.captureWorker.startRecord)
        self.captureThread.start()

        # self.ui.verticalLayout.addWidget(self.widget)

        listWidgetCustom = QCustomQWidget()
        listWidgetCustom.setTextUp("TEST")
        listWidgetCustom.setTextDown("Hello")
        print(self.dir)
        listWidgetCustom.setIcon(self.dir + '\MyIcon.png')
        myQListWidgetItem = QListWidgetItem(self.ui.listWidget)
        myQListWidgetItem.setSizeHint(listWidgetCustom.sizeHint())
        self.ui.listWidget.addItem(myQListWidgetItem)
        self.ui.listWidget.setItemWidget(myQListWidgetItem, listWidgetCustom)


class QCustomQWidget (QWidget):
    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__(parent)
        self.textQVBoxLayout = QVBoxLayout()
        self.textUpQLabel = QLabel()
        self.textDownQLabel = QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.textQVBoxLayout.addWidget(self.textDownQLabel)
        self.allQHBoxLayout = QHBoxLayout()
        self.iconQLabel = QLabel()
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        self.setLayout(self.allQHBoxLayout)
        # setStyleSheet
        self.textUpQLabel.setStyleSheet('''
            color: rgb(0, 0, 255);
        ''')
        self.textDownQLabel.setStyleSheet('''
            color: rgb(255, 0, 0);
        ''')

    def setTextUp(self, text):
        self.textUpQLabel.setText(text)

    def setTextDown(self, text):
        self.textDownQLabel.setText(text)

    def setIcon(self, imagePath):
        self.iconQLabel.setPixmap(QPixmap(imagePath).scaled(
            48, 48))


if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    main = MainWindow()
    main.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
