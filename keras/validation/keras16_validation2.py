from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 실습 자르기 (10:3:3)
# 1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


x_train = x[:10]
x_test = x[10:13]
x_validation = x[13:]
# x의 값을 3조각으로 나눈다.

y_train = y[:10]
y_test = y[10:13]
y_validation = y[13:]


# 2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mse',optimizer='adam', metrics=['mae','mse'])
model.fit(x_train,y_train,
          epochs=100,
          batch_size=1,
          validation_data=(x_validation, y_validation))

# 4.평가
loss = model.evaluate(x_test,y_test)
print('loss : ' , loss)

result = model.predict([17])
print('result : ', result)