from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import cv2
import random
import numpy as np


# MLP
feature_vector_length = 784

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], feature_vector_length)

input_shape =(feature_vector_length,)

X_train = X_train.astype('float32')

X_train /= 255

model = load_model("cnn.h5")

output = model.predict(X_train[0].reshape(1,feature_vector_length))

print("Model output :", output)
print("Value with highest probability :", np.argmax(output[0]))
