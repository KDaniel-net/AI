from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.정제된 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2.모델구성
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=10)

# 4.평가,예측
result = model.predict([6])
print('6의 예측값 : ' , result)

'''
6의 예측값 :  [[1.2942221]]
'''