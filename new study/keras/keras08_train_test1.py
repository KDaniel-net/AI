from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
# x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
# y = np.array(range(10))                 # (10, )
x_train = np.array([1,2,3,4,5,6,7])     # (7, )
x_test = np.array([8,9,10])             # (3, )
y_train = np.array(range(7))            # (7, )
y_test = np.array(range(7,10))          # (3, )

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

# 3. compile
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.predict
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

result = model.predict([10])
print('[10]의 결과:',result)

''' loss : 0.15513308346271515
[10]의 결과: [[8.825712]]'''

