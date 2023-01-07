from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])  
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

x = x.T
print(x.shape)
print(y.shape)

# 2.모델
model = Sequential()
model.add(Dense(6,input_dim=2))
model.add(Dense(4))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=20,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x,y)
print("정확도 : " , loss)

result = model.predict([[10,1.4]])
print("[10,1.4]의 값 : " , result)