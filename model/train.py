import os
import numpy as np
import time
from tensorflow.keras import callbacks, backend
import matplotlib.pyplot as plt
from model import unet
import cv2


TRAIN_PATH = 'data/'

train_ids = next(os.walk(TRAIN_PATH))[1]

X_train = np.zeros(shape=(294, 512, 512, 3))
Y_train = np.zeros(shape=(294, 512, 512, 3))
i = 0
j = 0

for id in train_ids:
    path = TRAIN_PATH + id
    imgs = os.listdir(path)
    print(path)
    print('i', i)
    for img in imgs:
        print(img)
        if img.find("scan") != -1:
            for k in range(j, j+6):
                print('k', k)
                y = cv2.imread(path+'/'+img)
                Y_train[k] = y/np.max(y)
        else:
            x = cv2.imread(path+'/'+img)
            X_train[i] = x/np.max(x)
            i+=1
    j+=6


backend.clear_session()
start_time = time.time()
model = unet(pretrained_weights='/checkpoint/best_weights_deep_model_n42.hdf5')
model.summary()
checkpoint_filepath = 'tmp/checkpoint/best_weights_deep_model_n112.hdf5'
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='mae',
    mode='min',
    save_best_only=True)
results = model.fit(X_train, Y_train, verbose=1, batch_size=4, epochs=100, callbacks=[model_checkpoint_callback])
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(results.history['mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.savefig('mae.png')

plt.figure()
plt.plot(results.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss.png')

plt.figure()
plt.plot(results.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('accuracy.png')