
# 실습
#1. R2를 음수가 아닌 0.5 이하로 줄이기
#2. 데이터는 건들지 말 것
#3. 레이어는 임풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 각각 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상
#8. loss지표는 mse 또는 mae
#9. activation 사용 금지
# [실습시작]
# 즉, R2를 강제적으로 나쁘게 만들어라.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import numpy as np

#1. data
x = np.array(range(1,21))
y = np.array(range(1,21))

x_train,x_test,y_train,y_test = train_test_split (x,y,
                                                  shuffle=True,
                                                  train_size=0.7,
                                                  random_state=123)

# 2. model
model = Sequential()
model.add(Dense(70,input_dim=1))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae','acc'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4. prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_absolute_error(y_test,y_predict))

print('RMSE :',RMSE(y_test,y_predict))

r2 = r2_score(y_test,y_predict)
print('R2 :',r2)

''' loss : [1.8154272529713467e-09, 4.0531158447265625e-05, 0.0]
RMSE : 0.006360163959173152
R2 : 0.9999999999196413 '''