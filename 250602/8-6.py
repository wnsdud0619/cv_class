import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# 사전 학습된 ResNet50 모델 로드
model = ResNet50(weights='imagenet')

# 이미지 로드 및 전처리
img = cv.imread('rabbit.jpg')  # 이미지 파일명
x = np.reshape(cv.resize(img, (224, 224)), (1, 224, 224, 3))  # 리사이즈 및 배치 차원 추가
x = preprocess_input(x)  # 모델에 맞는 전처리 적용

# 예측 수행
preds = model.predict(x)
top5 = decode_predictions(preds, top=5)[0]  # 상위 5개 결과

# 예측 결과 출력
print('예측 결과:', top5)

# 이미지에 예측 텍스트 오버레이
for i in range(5):
    label = top5[i][1] + ':' + str(top5[i][2])
    cv.putText(img, label, (10, 20 + i * 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 결과 이미지 출력
cv.imshow('Recognition result', img)
cv.waitKey(0)
cv.destroyAllWindows()
