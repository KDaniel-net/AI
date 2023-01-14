from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2.model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100)

# 4.prediction
result = model.predict([6])
print('결과 :',result)

''' Epoch 100/100
1/1 [==============================] - 0s 2ms/step - loss: 0.0304
결과 : [[6.0887175]]
 '''
''' 
좋은 결과값 가지기 : 중간계층(hidden layer)를 늘리거나 epochs의 값을 늘려준다. = 훈련횟수 증가

하이퍼파라미터 : 학습률, 신경망의 은닉층 갯수, 배치 사이즈, 에포크 수
하이퍼파라미터 튜닝 = 최적화 : 하이퍼파라미터의 값을 조정한다. '''