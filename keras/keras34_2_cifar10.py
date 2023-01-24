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
                 activation='relu'))
# filters = N장의 사진 kernel_size = 사진을 2*2로 자를 예정!! input_shape = 들어가는 이미지의 크기(RGB이미지를 가지고 있기 때문에 3을 적용해준다.)             
model.add(Conv2D(filters=64, kernel_size=(2,2)))         
model.add(Conv2D(filters=64, kernel_size=(2,2)))         
model.add(Flatten())
# 이미지 데이터를 펼친다고 생각하면 이해할수 있다. n차원을 -> 1차원으로                                   
model.add(Dense(50, activation='relu'))                
model.add(Dense(10, activation='softmax'))
# 다중 클래스 에서는 softmax만 사용이 가능하다. 

# 3.컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])              
model.fit(x_train, y_train, epochs=2, verbose=1, batch_size=32, validation_split=0.2)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5 ,restore_best_weights=True, verbose=1)
# 반복되는 횟수가 많을수록 오히려 과적합이 발생할수 있음.
# loss 값의 기준을 val_loss로 잡고, 해당값의 최소값이 5번 이상 변하지 않거나 커진다면 멈춰라
# 최적의 성능을 가진 값을 도출해 준다.
# https://deep-deep-deep.tistory.com/55

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
# 월(%m)일(%d)_시간(%H)분(%M) date값에 프로그래밍을 하는 날짜와 시간을 입력해준다.

filepath = './_save/mnist/cifar10/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k34_2_' + date + '_' + filename)
# monitor할 기준으로 최상의 값을 지정한 파일명으로 저장한느것

# 4.평가
results = model.evaluate(x_test, y_test)
print('loss : ', results[0], 'acc : ', results[1])
# metrics를 사용하여 출력값이 2개 이기 때문에 [0],[1]로 구분한다. 
# results[0] = sparse_categorical_crossentropy , results[1] = acc값이 출력된다.

