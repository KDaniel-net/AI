# [실습]
# R2= 0.55~0.6 이상

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# 1.데이터
dataset = fetch_california_housing()
x=dataset.data
y=dataset.target

print(x)
print(x.shape)      #(20640, 8)
print(y)
print(y.shape)      #(20640, )

print(dataset.feature_names)
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=123)

print(x_train.shape)        #(14447,8)

# 2. 모델 구성
model = Sequential()
model.add(Dense(6,input_dim=8))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print ('loss 값 : ' , loss )

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict))
#  mse값에 루트로 값을 출력하는것

r2 = r2_score(y_test, y_predict)
print(" R2 : ", r2)

