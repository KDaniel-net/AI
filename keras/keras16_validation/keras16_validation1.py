from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

# 검증 데이터를 추가한다.
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])

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