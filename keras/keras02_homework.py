from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)

# 13을 예측해 보세요.
# 1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2.모델구성
model = Sequential()
model.add(Dense(2, input_dim=1))

# 3.컴파일
model.compile(loss='mae' ,optimizer='adam')
model.fit(x,y,epochs=10)

# 4.평가
result =model.predict([13])
print(result)