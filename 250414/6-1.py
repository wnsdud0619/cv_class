from PyQt5.QtWidgets import *
import sys
import platform
import os

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기')
        self.setGeometry(200, 200, 500, 100)

        shortBeepButton = QPushButton('짧게 삑', self)
        longBeepButton = QPushButton('길게 삑', self)
        quitButton = QPushButton('나가기', self)

        shortBeepButton.move(20, 30)
        longBeepButton.move(120, 30)
        quitButton.move(220, 30)

        shortBeepButton.clicked.connect(self.shortBeep)
        longBeepButton.clicked.connect(self.longBeep)
        quitButton.clicked.connect(self.close)

    def shortBeep(self):
        self.beep(200)

    def longBeep(self):
        self.beep(1000)

    def beep(self, duration):
        system_name = platform.system()
        if system_name == "Windows":
            import winsound
            winsound.Beep(1000, duration)
        else:
            # 리눅스에서는 beep 명령어 사용 (패키지 설치 필요)
            os.system(f'beep -f 1000 -l {duration}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = BeepSound()
    win.show()
    sys.exit(app.exec_())
