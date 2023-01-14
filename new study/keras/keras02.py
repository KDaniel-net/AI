from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2.model
model = Sequential()
model.add(Dense(2,input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=5000)

# 3.prediction
result = model.predict([13])
print('결과 :' , result)

''' Epoch 5000/5000
1/1 [==============================] - 0s 7ms/step - loss: 0.0011
결과 : [[12.936455]] '''