from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 1.데이터
path = './_data/Bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',index_col=0)

print(train_csv)
print(train_csv.shape)
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

print(train_csv.info())
print(test_csv.info())
print(test_csv.describe())

# 결측치 처리 1.제거 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'],axis=1)
print(x) # [10886 rows x 8 columns]
y = train_csv['count']
# Name: count, Length: 10886, dtype: int64 // y에 count행을 만들어줌
print(y)
print(y.shape)  #(10886,) 위에서 count만 넣어주었기 때문에 10886행을 가지고
print(test_csv) # [6493 rows x 8 columns]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=5)
print(x_train.shape, x_test.shape) # (8164, 8) (2722, 8)
print(y_train.shape, y_test.shape) # (8164,) (2722,)

# 2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3.컴파일
model.compile(loss='mae',optimizer='adam',metrics=['mse'])
hist = model.fit(x_train,y_train,epochs=10,batch_size=400,
                 validation_split=0.2,
                 verbose=2)

# 4.평가
loss = model.evaluate(x_test,y_test)
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
