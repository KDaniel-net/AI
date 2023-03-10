import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# 경로를 일일이 치는것이 힘들기 때문에 path에 경로를 설정해주고 path를 이용하여 파일을 불러옴.
# index_col=0 : 0번째 column은 index로 데이터가 아님을 명시해주는 것이다. (여기에서는 id), 항상 컴퓨터는 0부터 시작.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)
print(train_csv.columns)

# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)
# trarin_set에 있는 마지막 count는 분리해줘야함.
print(x) # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) # (1459,)
print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1234)
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape) # (1021,) (438,)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=9))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print ('RMSE : ', rmse)

r2 = r2_score(y, y_predict)
print("R2 : ", r2)

y_submit = model.predict(test_csv)

# 결측치란? null 값을 의미한다. (trian_set.info())을 확인하면 non-null 값으로 null 값을 계산할 수 있다.
# 총 1459여야 하는데 1450만 있다면 9개가 null 값이다.