from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
# x = np.array([1, 2, 3, 4, 5, 6])
# y = np.array([1, 2, 3, 5, 4, 6])
x = np.array([1])
y = np.array([1])

# 2.모델
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=10,batch_size=40)

# 4.평가
loss = model.evaluate(x,y)
# 데이터를 평가해서 정확도를 나타내는것 
# evaluate의 값은 1에 가까울수록 정확하다는 의미
print('정확도 : ' , loss)

result = model.predict([6])
print("6의 예측값 : " , result)