from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# 1.데이터
dataset = fetch_california_housing()
x=dataset.data
y=dataset.target

print(x)
print(x.shape)      #(20640, 8)
print(y)
print(y.shape)      #(20640, )

print(dataset.feature_names)
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size=0.75, 
                                                    shuffle=True, 
                                                    random_state=123)

print(x_train.shape)        #(14447,8)

# 2. 모델 구성
model = Sequential()
model.add(Dense(6,input_dim=8))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train,y_train,epochs=100,batch_size=1,
                 validation_split=0.2, callbacks=[earlyStopping],
                 verbose=1)

# 4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

print('========================')
print(hist) #<keras.callbacks.History object at 0x0000016F21A30880>
print('========================')
print(hist.history) # hist안에 있는 리스트를 보여줌. history에는 loss값 val_loss값의 형태가 들어감
print('========================')
print(hist.history['loss'])    # hist 안에 있는 loss값만을 보고 싶다.
print('========================')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c='red', 
         marker='.', label ='loss')      # 점 형식으로 이름으 loss
plt.plot(hist.history['val_loss'],c='blue', 
         marker='.', label = 'val_loss')
plt.grid()                          # 그래프에 격자 추가
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()                        # 범주 만들어줌
# plt.legend(loc='upper left')        # 범주의 위치
plt.show()

'''



'''