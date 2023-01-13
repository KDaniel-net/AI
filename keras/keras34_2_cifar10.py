from tensorflow.keras.datasets import cifar10,cifar100
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)

print(x_train.shape)                    # (50000, 32, 32, 3)
print(x_test.shape)                     # (10000, 32, 32, 3)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# 2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32,32,3), 
                 activation='relu'))                    # (31,31,128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))         # (30,30,64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))         # (29,29,64) -> 53824
model.add(Flatten())                                    # -> 53824
model.add(Dense(32, activation='relu'))                 # input_shape = (53824,)
model.add(Dense(10, activation='softmax'))

# 3.컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])              # metrics에 acc가 적용이 되었기 때문에 results에는 2개의 값이 들어간다.

model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=100, validation_split=0.2)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5 ,restore_best_weights=True, verbose=1)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# filepath = './_save/mnist/cifar10'
# filename = '{epoch:02d}-{val_loss:.2f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k34_2_' + date + '_' + filename)

# 4.평가
results = model.evaluate(x_test, y_test)
print('loss : ', results[0], 'acc : ', results[1])

