from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import align.detect_face
import tensorflow as tf
import numpy as np
import os


class RegisterThreading(QObject):

    # directory of folder
    dir = os.path.dirname(os.path.realpath(__file__))

    # Signal & Slot
    imagePayload = pyqtSignal(np.ndarray)
    finishThread = pyqtSignal()
    startRecord = pyqtSignal(str)
    # Flag

    finishFlag = False
    recordFlag = False

    # Variable

    nameRegistor = ""
    countImage = 0

    def __init__(self, parent=None):
        super(RegisterThreading, self).__init__(parent=parent)

        # Signal
        self.finishThread.connect(self.finished)

        self.startRecord.connect(self.handleRecord)

    def startRegistor(self):
        npy = './align'
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 183
        input_image_size = 160

        # Initial Graph
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                cap = cv2.VideoCapture(0)
                print('Start Recognition')
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, npy)
                while True:
                    ret, frame = cap.read()

                    frame = cv2.resize(frame, (460, 350), fx=0.5, fy=0.5)

                    bounding_boxes, points = align.detect_face.detect_face(
                        frame, minsize, pnet, rnet, onet, threshold, factor)
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if ret:

                        for i in range(len(bounding_boxes)):
                            # Record image
                            if self.recordFlag and self.nameRegistor != "" and self.countImage <= 30:
                                cv2.imwrite(
                                    self.dir+"/imageTest/{}/{}.jpg".format
                                    (self.nameRegistor, int(self.countImage)), frame)
                                self.countImage += 1

                            # Draw Rectangle
                            cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (
                                int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (0, 255, 0), 2)

                        # Draw text
                        cv2.putText(frame, "Count : {}/30".format(self.countImage), (50, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)

                        # Draw point noise,eye
                        for p in points.T:
                            for x in range(5):
                                cv2.circle(
                                    frame, (p[x], p[x+5]), 1, (0, 255, 0), 2)

                    # Emit Data to Widget
                    self.imagePayload.emit(frame)

                    if self.finishFlag:
                        break

        cap.release()
        cv2.destroyAllWindows()

    def finished(self):
        self.finishFlag = True

    def handleRecord(self, name):
        self.recordFlag = True
        self.nameRegistor = name
        self.countImage = 0
