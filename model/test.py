import os
import cv2
import numpy as np
from model import unet

TEST_PATH = 'test/'
SAVE_PATH = 'res/'
TRAIN_PATH = 'data/resized2_0027/'

model = unet(pretrained_weights='/checkpoint/best_weights_deep_model_n112.hdf5')

imgs = os.listdir(TEST_PATH)

for img in imgs:
    X_test_list = []
    x = cv2.imread(TEST_PATH + img)
    X_test_list.append(x/np.max(x))
    X_test = np.array(X_test_list)
    test_pred = model.predict(X_test, verbose=1)
    output = test_pred[0] * np.max(x)
    output[output<0] = 0
    output[output>255] = 255
    output = np.uint8(output)
    cv2.imwrite(SAVE_PATH + img[:-4] + '_res_test.png', output)

imgs = os.listdir(TRAIN_PATH)

for img in imgs:
    X_test_list = []
    x = cv2.imread(TRAIN_PATH + img)
    X_test_list.append(x/np.max(x))
    X_test = np.array(X_test_list)
    test_pred = model.predict(X_test, verbose=1)
    output = test_pred[0] * np.max(x)
    output[output<0] = 0
    output[output>255] = 255
    output = np.uint8(output)
    cv2.imwrite(SAVE_PATH + img[:-4] + '_res_train.png', output)