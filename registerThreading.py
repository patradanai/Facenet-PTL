from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import align.detect_face
import tensorflow as tf
import numpy as np


class RegisterThreading(QObject):

    # Signal & Slot
    imagePayload = pyqtSignal(np.ndarray)
    finishThread = pyqtSignal()

    # Flag

    finishFlag = False

    def __init__(self, parent=None):
        super(RegisterThreading, self).__init__(parent=parent)

        # Signal
        self.finishThread.connect(self.finished)

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
                            cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (
                                int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (0, 255, 0), 2)

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
