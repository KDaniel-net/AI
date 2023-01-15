from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array([range(10),range(21,31),range(201,211)]) # (3, 10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]) # (2, 10)

# [실습] train_test_split을 이용하여
# 7:3으로 잘라서 모델 구현 / 소스 완성하기

x = x.T
y = y.T

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=123)

print(x_train, y_train)
print(x_test, y_test)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=3))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2))

# 3. compile
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=1)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 결과값 :',result)

''' loss : 0.3885127007961273
[9,30,210]의 결과값 : [[9.633979  2.2566054]] '''