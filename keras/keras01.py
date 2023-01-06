import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

print(x)
# 2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
# [1,2,3]은 한 덩어리로 dim값을 1만 줌

# 3.컴파일
model.compile(loss='mae' , optimizer='adam')
model.fit(x,y,epochs=100)

# 4.평가,예측
result = model.predict([1,2,3])
print(result)

