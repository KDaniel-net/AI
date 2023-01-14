from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)

# 1.data

x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. model
model = Sequential()
model.add(Dense(1, input_dim=1))
# 한개를 출력하고 입력을 한개 한다.

# 3. compile
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=400)
# 모델을 학습시키는 함수. 필수 인자로는 3가지가 있다. (훈련 데이터, 정답 데이터, 반복횟수)

# 4.prediction
result = model.predict([500])
# 입력한 값을 예측해서 출력해 준다. // epochs가 클수록 학습이 많이 되었기 때문에 정확도가 올라간다.
print('결과 : ', result)

''' Epoch 1000/1000
1/1 [==============================] - 0s 4ms/step - loss: 3.6271e-04
결과 :  [[500.03094]] 

Epoch 400/400
1/1 [==============================] - 0s 3ms/step - loss: 0.0261
결과 :  [[480.6443]]
'''