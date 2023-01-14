from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,6,5])

# 2.model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=2)
# batch_size : 몇개의 단위 사이즈로 자를지에 대한 설명이다.
# 예) 1,2,3,4,5,6 -> 1,2 3,4 5,6

# 4.prediction
result = model.predict([6])
print('결과 :',result)

''' Epoch 100/100           
1/1 [==============================] - 0s 5ms/step - loss: 0.3352
결과 : [[5.882936]] '''