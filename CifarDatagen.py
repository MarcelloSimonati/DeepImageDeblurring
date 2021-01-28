
from tensorflow.keras.datasets import cifar10
import numpy as np
import random
import cv2

def cifar_datagen(random_seed):
    random.seed(random_seed)

    #Download Dataset
    (x_train, _), (x_test, _) = cifar10.load_data()
    kdim = (0,0)
    x_train_blur = []

    #Blurring Training Set
    for img in x_train:
        stdev = random.random() * 3.0
        dst = cv2.GaussianBlur(img ,kdim, sigmaX = stdev, sigmaY = stdev, borderType = cv2.BORDER_DEFAULT)
        x_train_blur.append(dst)

    x_train_blur = np.array(x_train_blur)

    #Blurring Test Set
    x_test_blur = []
    for img in x_test:
        stdev = random.random() * 3.0
        dst = cv2.GaussianBlur(img ,kdim, sigmaX = stdev, sigmaY = stdev, borderType = cv2.BORDER_DEFAULT)
        x_test_blur.append(dst)

    x_test_blur = np.array(x_test_blur)

    #Image Normalization
    x_train = x_train / 255.0
    x_train_blur = x_train_blur / 255.0
    x_test = x_test / 255.0
    x_test_blur = x_test_blur / 255.0

    return (x_train, x_train_blur), (x_test, x_test_blur)