from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([2,4,6,8,10,12,14,16,18,20])
x = x.T
print(x.shape, y.shape) # (10, 3) (10,)

# 2.model
model = Sequential()
model.add(Dense(40,input_dim=3))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

# 4. prediction
loss = model.evaluate(x,y)
print('loss :',loss)

result = model.predict([[10,1.4,0]])
print('[10, 1.4, 0]의 결과값 : ', result)

''' loss : 0.11955733597278595
[10, 1.4, 0]의 결과값 :  [[20.18592]] '''