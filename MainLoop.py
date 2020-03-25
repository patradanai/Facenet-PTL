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
from WidgetList import *
from WidgetCustomize import *


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

    # Flag

    flagPreTraining = True
    flagFaceDetection = True

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
        # for State - Stop of Pre-Training
        self.ui.pushButton_11.setText('START')
        self.ui.pushButton_9.setText('START')

        # Init Button Disable
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_6.setEnabled(False)

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
        self.registerThread.start()

        # Stack Widgetr

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
        self.ui.pushButton_11.clicked.connect(self.openCameraPreTraining)

        # ------------------ Face-Recognition -------------
        self.ui.pushButton_7.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentIndex(2))   # Back to MainPage
        self.ui.pushButton_9.clicked.connect(
            self.openCameraFaceDetect)  # Start Face

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

    # ------------ Function for Face Detection ------------

    def openCameraFaceDetect(self):
        self.flagFaceDetection = not self.flagFaceDetection

        if self.flagFaceDetection == False:
            self.ui.pushButton_9.setText('STOP')
            # Set Disable Button Back
            self.ui.pushButton_10.setEnabled(False)
        else:
            self.ui.pushButton_9.setText('START')
            # Set Disable Button Back
            self.ui.pushButton_10.setEnabled(True)

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

    def trainingModel(self):
        self.registerWorking.finishThread.emit()
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
                sys.exit("All Done")
                state += 1
            else:
                break

    def openCameraPreTraining(self):
        self.flagPreTraining = not self.flagPreTraining

        if self.flagPreTraining == False:
            self.ui.pushButton_11.setText('STOP')
            # Set Disable Button Back
            self.ui.pushButton_7.setEnabled(False)
            self.registerWorking.startCamera.emit()
        else:
            self.ui.pushButton_11.setText('START')
            # Set Disable Button Back
            self.ui.pushButton_7.setEnabled(True)
            self.registerWorking.startCamera.emit()


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
