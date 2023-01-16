from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1. data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=123)

# 2. model
model = Sequential()
model.add(Dense(40,input_dim=1))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae', 'mse', 'acc'])
''' metrics는 2개 이상의 값을 입력할수 있다. 
accuracy = acc, 평균 절대 오차, 평균 제곱 오차, 정확도를 의미하고 있다. '''
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4. prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

''' 14.9034 - mae: 3.0339 - mse: 14.9034 - acc: 0.0000e+00
loss : [14.9033842086792, 3.0339107513427734, 14.9033842086792, 0.0]
 '''