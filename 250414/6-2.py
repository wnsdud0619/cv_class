import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

import sys
import cv2 as cv
import numpy as np

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200, 200, 800, 600)

        # 비디오 창용 라벨
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 640, 480)

        # 버튼들
        videoButton = QPushButton('비디오 켜기', self)
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(120, 10, 100, 30)
        saveButton.setGeometry(230, 10, 100, 30)
        quitButton.setGeometry(340, 10, 100, 30)

        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            self.close()
        self.timer.start(30)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

    def captureFunction(self):
        if hasattr(self, 'frame'):
            self.capturedFrame = self.frame.copy()
            QMessageBox.information(self, "프레임 캡처", "프레임을 캡처했습니다!")

    def saveFunction(self):
        if hasattr(self, 'capturedFrame'):
            fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './frame.jpg', "Images (*.png *.jpg)")
            if fname:
                cv.imwrite(fname, self.capturedFrame)
                QMessageBox.information(self, "저장 성공", f"{fname} 로 저장되었습니다.")

    def quitFunction(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.timer.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Video()
    win.show()
    sys.exit(app.exec_())
