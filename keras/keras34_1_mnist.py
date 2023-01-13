import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(x_train.shape)                    # (60000, 28, 28, 1)
print(x_test.shape)                     # (10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# 2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28,28,1), 
                 activation='relu'))                    # (27,27,128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))         # (26,26,64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))         # (25,25,64) -> 40000
model.add(Flatten())                                    # -> 40000
model.add(Dense(32, activation='relu'))                 # input_shape = (40000,)
                                                        # (6만, 4만)인풋 (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

# 3.컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])              # metrics에 acc가 적용이 되었기 때문에 results에는 2개의 값이 들어간다.

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5 ,restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_saver/mnist'
filename = '{epoch:02d}-{val_loss:.2f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k31_10_' + date + '_' + filename)

# 4.평가
results = model.evaluate(x_test, y_test)
print('loss : ', results[0], 'acc : ', results[1])

