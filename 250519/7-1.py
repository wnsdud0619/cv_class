import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # ①

plt.figure(figsize=(24, 3))
plt.suptitle('MNIST', fontsize=30)

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')  # ②
    plt.xticks([])
    plt.yticks([])
    plt.title(str(y_train[i]), fontsize=30)

plt.show()
