from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#1. data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split (x, y,
                                                  shuffle=True,
                                                  train_size=0.7,
                                                  random_state=123)

# 2.model
model = Sequential()
model.add(Dense(40,input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['acc','mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

print("================================")
print(y_test)
print(y_predict)
print("================================")

def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE : ", RMSE(y_test, y_predict))

# RMSE :  3.8413956502397646