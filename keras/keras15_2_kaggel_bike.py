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
model.fit(x_train,y_train,epochs=3000,batch_size=400)

# 4.평가
loss = model.evaluate(x_test,y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit


submission_csv.to_csv(path + 'submission_0108.csv')
print('REMS : ', rmse)
print('R2 :' , r2)
