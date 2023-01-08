# 실습
# R2 0.62이상

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(442, 10)
print(y)
print(y.shape)  #(442,)

print(dataset.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(dataset.DESCR)
# 사이킥런의 데이터 요약

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75, shuffle=True, random_state=123
                                                    )
print(x_train.shape)    #(331, 10)

# 2.모델구성
model = Sequential()
model.add(Dense(80, input_dim=10))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=8)

# 4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict))
#  mse값에 루트로 값을 출력하는것

r2 = r2_score(y_test, y_predict)
print(" R2 : ", r2)

'''
Epoch 150/150
331/331 [==============================] - 1s 2ms/step - loss: 3005.8962 - mae: 44.1791
4/4 [==============================] - 0s 0s/step - loss: 2917.4890 - mae: 43.9745
loss : [2917.489013671875, 43.97453308105469]
RMSE : 54.013788027789516
 R2 :  0.509292214788622
 
 
 
'''
