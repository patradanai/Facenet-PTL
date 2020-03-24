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
from preprocessing import *
from classifier import *


class MainWindow(QMainWindow):

    # directory of folder
    dir = os.path.dirname(os.path.realpath(__file__))

    # For Make image Pre-process
    input_datadir = './imageTest'
    output_datadir = './imageTrain'

    # For Make Classifire
    datadir = './imageTrain'
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'

    # Variable

    listName = []
    listFolder = []

    nameFolder = ""

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

        # Init Button Disable
        self.ui.pushButton_5.setEnabled(False)
        # self.ui.pushButton_6.setEnabled(False)

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

        # Face Registor
        self.registerThread = QThread()
        self.registerWorking = RegisterThreading()
        self.registerWorking.moveToThread(self.registerThread)
        self.registerWorking.imagePayload.connect(
            self.ui.widget_2.image_data_slot)
        self.registerThread.started.connect(self.registerWorking.startRegistor)
        # self.registerThread.start()

        # Stack Widget

        # ------------------ MainPage ---------------------
        self.ui.pushButton.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(0))   # Go to Face recognition
        self.ui.pushButton_2.clicked.connect(
            self.goToPreImage)   # Go to Pre-image
        self.ui.pushButton_3.clicked.connect(sys.exit)          # Exit

        # ------------------ Pre-Training -----------------
        self.ui.pushButton_10.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(2))   # Back to MainPage
        self.ui.pushButton_4.clicked.connect(self.createName)   # Add Folder
        self.ui.pushButton_5.clicked.connect(
            lambda: self.registerWorking.startRecord.emit(self.nameFolder))   # Start Record

        # ------------------ Face-Recognition -------------
        self.ui.pushButton_7.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(2))   # Back to MainPage
        self.ui.pushButton_9.clicked.connect(
            self.captureThread.start)  # Start Face
        self.ui.pushButton_8.clicked.connect(
            self.captureThread.finished)  # Stop Face

        # ListName in Folder
        print(os.listdir(self.dir + "/imageTest/"))

        self.handlelistFolder(os.listdir(self.dir + "/imageTest/"))

        self.ui.pushButton_6.clicked.connect(self.trainingModel)

        self.ui.pushButton_7.clicked.connect(
            self.registerWorking.finishThread.emit)

    def handleList(self, list):

        if not list in self.listName:
            self.listName.append(list)

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

    def handlelistFolder(self, list):
        self.listFolder = list

        # Render image in Qlistwidget

        if len(self.listFolder) > 0:
            for i in self.listFolder:
                listWidgetCustom = QListQWidget()
                listWidgetCustom.setTextUp(i)
                listWidgetCustom.setTextDown("Date : ")
                listWidgetCustom.setIcon(
                    self.dir + "/imageShow/" + i + ".jpg")
                myQListWidgetItem = QListWidgetItem(self.ui.listWidget_2)
                myQListWidgetItem.setSizeHint(listWidgetCustom.sizeHint())
                self.ui.listWidget_2.addItem(myQListWidgetItem)
                self.ui.listWidget_2.setItemWidget(
                    myQListWidgetItem, listWidgetCustom)

    # ------------ Function for Pre-Training --------------
    def createFolder(self):
        if not os.path.exists(self.dir + "/imageTest/" + "%s" % self.nameFolder):
            os.makedirs(self.dir + "/imageTest/" + "%s" % self.nameFolder)

    def createName(self):
        self.nameFolder = ""
        text, okPressed = QInputDialog.getText(



            self, "Get text", "Your name:", QLineEdit.Normal, "")
        if okPressed and text != '':
            self.nameFolder = text
            self.createFolder()
            # Boolean Button
            self.ui.pushButton_4.setEnabled(False)
            self.ui.pushButton_5.setEnabled(True)

    def goToPreImage(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.registerThread.start()

    def trainingModel(self):
        state = 0
        while True:
            if state == 0:
                # Pre-process
                obj = preprocessing(self.input_datadir, self.output_datadir)
                nrof_images_total, nrof_successfully_aligned = obj.alignProcessing()
                print('Total number of images: %d' % nrof_images_total)
                print('Number of successfully aligned images: %d' %
                      nrof_successfully_aligned)
                state += 1
                # Classifier
            elif state == 1:
                print("Training Start")
                objModel = classifier(mode='TRAIN', datadir=self.datadir, modeldir=self.modeldir,
                                      classifierFilename=self.classifier_filename)
                get_file = objModel.main()
                print('Saved classifier model to file "%s"' % get_file)
                sys.exit("All Done")
                state += 1
            else:
                break
# Widgetlist Name


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

# Widgetlist Folder


class QListQWidget (QWidget):
    def __init__(self, parent=None):
        super(QListQWidget, self).__init__(parent)
        self.textQVBoxLayout = QVBoxLayout()
        self.textUpQLabel = QLabel()
        self.textDownQLabel = QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)

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
