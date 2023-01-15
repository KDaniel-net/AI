from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array([range(10),range(21,31),range(201,211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

x = x.T
y = y.T

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=3))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(2))

# 3. compile
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=300,batch_size=2)

# 4. prediction
loss = model.evaluate(x,y)
print('loss :',loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 값은 : ', result)

''' loss : 0.48240455985069275
[9,30,210]의 값은 :  [[9.311862  1.2849066]] '''