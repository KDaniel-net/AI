from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])       # (7, )
x_test = np.array([8, 9, 10])                   # (3, )
y_train = np.array(range(7))                    # (7, )  [0,1,2,3,4,5,6] // range(7) : 0 ~ 7-1
y_test = np.array(range(7, 10))                 # (3, )  [7,8,9]  // range(7,10) : 7 ~ 10-1

# 2.모델
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=20)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ' , loss)

result = model.predict([[9]])
print('[9]의 예측값 : ' , result)

'''
loss :  0.8902807235717773...20
[9]의 예측값 :  [[7.109719]]

loss :  0.4512254297733307...30
[9]의 예측값 :  [[7.548775]]
'''