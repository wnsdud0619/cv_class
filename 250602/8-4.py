import tensorflow as tf
import numpy as np
import cv2

# 모델 로드 (CNN 버전)
model = tf.keras.models.load_model('cnn_v2.h5')

# 손글씨 이미지를 전처리하는 예시 함수 (사용자 입력 이미지에 맞게 수정 필요)
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28))
    img_normalized = img_resized / 255.0
    return img_normalized.reshape(1, 28, 28, 1)

# 예시 숫자 이미지들 (여기선 5개 숫자를 입력받았다고 가정)
# 예를 들어 사용자가 그린 숫자 이미지를 5개 추출했다면:
numerals = np.array([img1, img2, img3, img4, img5])  # 각 img는 28x28 크기의 numpy 배열

# 전처리
numerals = numerals.reshape(5, 28, 28, 1)

# 예측
results = model.predict(numerals)
predicted_digits = np.argmax(results, axis=1)

# 결과 출력
print("인식된 숫자:", predicted_digits)
