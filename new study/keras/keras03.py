from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2.model
model = Sequential()
model.add(Dense(1,input_dim=1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4.prediction
result = model.predict([16])
print('결과 :',result)

''' Epoch 1000/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.4001
결과 : [[15.998298]] '''