import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        # 버튼 및 라벨 구성
        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러옴', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        # 버튼 위치 설정
        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        # 버튼 기능 연결
        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 표지판 데이터 초기화
        self.signFiles = [['./ch6img/child.png', '어린이'], ['./ch6img/elder.png', '노인'], ['./ch6img/disabled.png', '장애인']]
        self.signImgs = []
        self.roadImg = None

    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')

        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname, self.signImgs[-1])

    def roadFunction(self):
        if self.signImgs == []:
            self.label.setText('먼저 표지판을 등록하세요.')
        else:
            fname = QFileDialog.getOpenFileName(self, '파일 읽기', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('파일을 찾을 수 없습니다.')
            cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
        else:
            sift = cv.SIFT_create()

            # 표지판 영상의 키포인트와 디스크립터
            KD = []
            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))

            # 도로 영상의 키포인트와 디스크립터
            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)

            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            GM = []

            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, 2)
                T = 0.7
                good_match = []
                for nearest1, nearest2 in knn_match:
                    if nearest1.distance / nearest2.distance < T:
                        good_match.append(nearest1)
                GM.append(good_match)

            best = GM.index(max(GM, key=len))
            img_match = cv.drawMatches(self.signImgs[best], KD[best][0], self.roadImg, road_kp, GM[best], None)
            cv.imshow('Matches and Homography', img_match)

            # 보호 구역 알림
            self.label.setText(self.signFiles[best][1] + ' 보호구역입니다. 30km로 서행하세요.')
            winsound.Beep(3000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

# 실행
app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()
