from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape, y.shape) # (2, 10) (10,)
# x,y의 배열을 출력해준다. 

x = x.T
# 행을 맞춰주어야 하기 때문에 x를 변환해준다.
print(x.shape, y.shape) # (10, 2) (10,)

# 2.model
model = Sequential()
model.add(Dense(50,input_dim=2))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=2)

# 4.prediction
loss = model.evaluate(x,y)
print('loss :',loss)

result = model.predict([[10,1.4]])
print('[10,1.4]의 값:', result)

''' mlp는 multiple을 의미한다. 다중 입력값을 의미한다.
입력 값이 2개 이상일때는 []로 묶어 주어야 한다. 예) [[1,3],[8,9]]

열 = fiture, column, 특성 '''