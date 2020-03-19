import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui_Widget import Ui_Form
import cv2
import numpy as np


class VideoDisplayWidget (QWidget):
    def __init__(self, parent=None):
        super(VideoDisplayWidget, self).__init__(parent)
        self.image = QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    @pyqtSlot(np.ndarray)
    def image_data_slot(self, image_data):
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    @pyqtSlot(np.ndarray)
    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()
