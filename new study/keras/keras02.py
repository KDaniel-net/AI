from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2.model
model = Sequential()
model.add(Dense(1,input_dim=1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=5000)

# 3.prediction
result = model.predict([13])
print('결과 :' , result)

''' Epoch 5000/5000
1/1 [==============================] - 0s 2ms/step - loss: 5.8052e-04
결과 : [[13.001655]] '''