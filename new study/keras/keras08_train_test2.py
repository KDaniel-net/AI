from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array(range(10))

# 실습 : 넘파이 리스트 슬라이싱으로 7:3으로 자르기!!
x_train = x[:-3]
x_test = x[-3:]
# 앞에서 -3까지 하고 : 마지막까지 정의한다. 즉, 8,9,10을 출력한다.
y_train = y[:7]
y_test = y[7:]

print(x_train,y_train)  # [1 2 3 4 5 6 7] [0 1 2 3 4 5 6]
print(x_test,y_test)    # [ 8  9 10] [7 8 9]

# 2. model
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

result = model.predict([11])
print('[11]의 결과 : ', result)

''' loss : 0.12514559924602509
[11]의 결과 :  [[10.14716]] '''