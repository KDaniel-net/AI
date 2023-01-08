# 실습
# r2를 음수가아닌 0.5이하로 줄이기
# 1.데이터는 건들지 말것.
# 3.레이어는 인풋 아웃풋 포함 7개 이상.
# 4. batch_size=1
# 5.히든레이어의 노드는 각각 10개이상 100개이하
# 6.train 70%
# 7. epoch 100번이상
# 8.loss 지표는 mse 또는 mae
# 9.activation사용금지
# [실습 시작]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

# 1.데이터
x = np.array(range(1, 21))
y = np.array(range(21, 41))


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=123)
# 2.모델구성
model = Sequential()
model.add(Dense(100,input_dim=1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss :' ,loss)

y_predict = model.predict(x_test)

def RMSE(y_test,x_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :' , RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)