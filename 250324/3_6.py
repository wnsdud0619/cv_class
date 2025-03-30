import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('/home/jykim/cv_class/soccer.jpg')

# 명암 영상으로 변환하고 출력
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# 히스토그램을 구해 출력
h = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1), plt.show()

# 히스토그램을 평활화하고 출력
equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# 히스토그램을 구해 출력
h = cv.calcHist([equal], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1), plt.show()
