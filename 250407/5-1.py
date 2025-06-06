import cv2 as cv
import numpy as np

img = np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,1,1,0,0,0,0,0,0],
                [0,0,1,1,1,0,0,0,0,0],
                [0,0,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)

ux = np.array([[-1,0,1]])
uy = np.array([[-1],[0],[1]])
k = cv.getGaussianKernel(3,1)
g = np.outer(k,k.transpose())

dy = cv.filter2D(img, cv.CV_32F, uy)
dx = cv.filter2D(img, cv.CV_32F, ux)
dyy = dy * dy
dxx = dx * dx
dyx = dy * dx
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)

C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) * (gdyy + gdxx)

for j in range(1, C.shape[0]-1): 
    for i in range(1, C.shape[1]-1):
        if C[j,i] > 0.1 and sum(sum(C[j-1:j+2,i-1:i+2])) == 8 * C[j,i]:
            img[j,i] = 9

np.set_printoptions(precision=2)
print(dy)     # ①
print(dx)     # ②
print(dyy)    # ③
print(dxx)    # ④
print(dyx)    # ⑤
print(gdyy)   # ⑥
print(gdxx)   # ⑦
print(gdyx)   # ⑧
print(C)      # ⑨
print(img)    # ⑩

popping = np.zeros([160,160], np.uint8)
for j in range(0,160):
    for i in range(0,160):
        popping[j,i] = np.uint8((C[j//16, i//16] + 0.06) * 700)

cv.imshow('Image Display2', popping)  # ⑪
cv.waitKey()
cv.destroyAllWindows()
