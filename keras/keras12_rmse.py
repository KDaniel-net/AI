from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=123)

# 2.모델구성
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)

print('-----------')
print(y_test)
print('-----------')
print(y_predict)
print('-----------')

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
#변수 두개 받아서 mse 처리=> 제곱근 값으로 리턴


print("RMSE :", RMSE(y_test, y_predict))