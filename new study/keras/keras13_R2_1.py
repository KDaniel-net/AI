from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

#1. data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,
                                                 train_size=0.7,
                                                 random_state=123)

# 2. model
model = Sequential()
model.add(Dense(40,input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae','acc'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

print("================================")
print(y_test)
print(y_predict)
print("================================")

def RMSE(y_test,y_predict):
    return np.sqrt(mean_absolute_error(y_test,y_predict))

print('RMSE :',RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict) 
# 모델에 대하여 잘 설명하고 있는지에 대하여 측정한 점수이다.
# 0~1의 사이값을 가지며, 1에 가까울소록 모델 설명된다는 것을 의미한다.

print("R2 : ", r2)

''' 
RMSE : 1.7419794753887834
R2 :  0.6464835707589486 
'''