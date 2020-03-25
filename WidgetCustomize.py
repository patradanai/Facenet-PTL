from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

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
