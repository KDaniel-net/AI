from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 2.model
model = Sequential()
model.add(Dense(800,input_dim=1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))    
model.add(Dense(1))    

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100)

# 4.prediction
result = model.predict([16])
print('결과 :',result)

# Epoch 100/100
# 1/1 [==============================] - 0s 4ms/step - loss: 0.4161
# 결과 : [[15.256533]]