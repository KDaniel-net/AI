from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd

# 1. data
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae','acc'])
model.fit(x_train,y_train,epochs=100,batch_size=1,
          validation_data=(x_validation, y_validation))

#4. prediction
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('[17]의 값 : ', result)

'''
loss :  [7.49444836856128e-07, 0.0008557637338526547, 0.0]
[17]의 값 :  [[16.998339]]
'''