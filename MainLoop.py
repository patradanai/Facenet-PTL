import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui_Widget import Ui_Form
import cv2
import numpy as np
from ui_DialogMainLoop import Ui_MainWindow
from captureThreading import *
from registerThreading import *
import qdarkstyle


class MainWindow(QMainWindow):

    # directory of folder
    dir = os.path.dirname(os.path.realpath(__file__))

    listName = []

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Face_Recognition")
        self.ui.label.setText("Face Recognition Check Name")

        # Init Set Text
        self.ui.pushButton.setText('Face Recognition')
        self.ui.pushButton_2.setText('Registor Face')
        self.ui.pushButton_3.setText('Exit')

        # Init Stack Widget
        self.ui.stackedWidget.setCurrentIndex(2)

        # Init Threading

        # Face Recognition
        self.captureThread = QThread()
        self.captureWorker = RecordVideo()
        self.captureWorker.moveToThread(self.captureThread)
        self.captureWorker.imageData.connect(self.ui.widget.image_data_slot)
        self.captureWorker.imageList.connect(self.handleList)
        self.captureThread.started.connect(self.captureWorker.startRecord)
        # self.captureThread.start()

        # Face Registor
        self.registerThread = QThread()
        self.registerWorking = RegisterThreading()
        self.registerWorking.moveToThread(self.registerThread)
        self.registerWorking.imagePayload.connect(
            self.ui.widget_2.image_data_slot)
        self.registerThread.started.connect(self.registerWorking.startRegistor)
        self.registerThread.start()

        # Stack Widget

        self.ui.pushButton.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_2.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_3.clicked.connect(sys.exit)

    def handleList(self, list):
        self.listName = list

        # Render image in Qlistwidget

        if len(self.listName) > 0:
            for i in self.listName:
                listWidgetCustom = QCustomQWidget()
                listWidgetCustom.setTextUp("PATRADANAI NAKPIMAY")
                listWidgetCustom.setTextDown("Date : ")
                listWidgetCustom.setIcon(
                    self.dir + "/imageShow/" + i + ".jpg")
                myQListWidgetItem = QListWidgetItem(self.ui.listWidget)
                myQListWidgetItem.setSizeHint(listWidgetCustom.sizeHint())
                self.ui.listWidget.addItem(myQListWidgetItem)
                self.ui.listWidget.setItemWidget(
                    myQListWidgetItem, listWidgetCustom)


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
    # setup stylesheet
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # Create and show the form
    main = MainWindow()
    main.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
