from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])

x = x.T
y = y.T

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(3))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

# 4.prediction
loss = model.evaluate(x,y)
print('loss :',loss)

result = model.predict([[9]])
print('[9]의 값은 : ', result)

''' loss : 0.13715651631355286
[9]의 값은 :  [[10.032667    1.6866956   0.46280023]] '''

# 평가 데이터와 훈련데이터가 같기 때문에 좋은 데이터라고 말할수는 없다. 
# : 평가할때는 다른값을 이용해서 할것, 데이터를 더 쪼갤것 ex)validation