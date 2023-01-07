from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])

x = x.T
y = y.T
print(x)
# 2.모델
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(5))
model.add(Dense(2))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=2)

# 4.평가,예측
loss = model.evaluate(x,y)
# evaluate의 값은 1에 가까울수록 좋다.
print('정확도 : ' , loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 예측값 : ', result)

'''
정확도 :  0.37443310022354126
[9,30,210]의 예측값 :  [[9.512758  1.5989405]]
'''