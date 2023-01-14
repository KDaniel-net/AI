from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2. model
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=2)

# 4.prediction
loss = model.evaluate(x,y)
# loss는 모델이 학습 데이터를 예측을 얼마나 잘 하는지에 대한 측정한 척도를 말한다.
print('loss:',loss)

result = model.predict([7])
print('결과:',result)

''' loss: 0.44342413544654846
결과: [[6.562006]] '''


# evaluate : 평가하는데 사용되는 메서드 이다. accuracy, loss값 등을 사용하여 측정한다.
# loss 값은 0~1까지의 값이 반환이 되고 0과 가까울수록 더 좋은 값이라고 할수 있다.
# 결과는 predict의 값으로 측정하는 것이 아니라 loss의 값으로 좋다 나쁘다를 말할수 있다.