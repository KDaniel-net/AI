from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([0,1,2,3,4,5,6,7,8,9])   # 10,
y = np.array ([range(10),
               [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])     #3,10
# 행을 같이 맞추어 주어야 하기 때문에 치환이 필요
y = y.T

print(x.shape)  # (10,)
print(y.shape)  # (10,3) // 변환을 해줌.

# 2.모델구성
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(3))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x,y)
print('loss : ' , loss)

result = model.predict([[9]])
print('[9]의 예측값 : ', result)

'''
loss :  1.1143866777420044
[9]의 예측값 :  [[9.042032  1.5877768 2.6751657]]
'''