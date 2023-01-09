import numpy as np #  숫자 연산
import pandas as pd # 데이터 등 다양한 형태로 들어가있는 상태
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
start = time.time()
# 1.데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)  # index = 데이터가 아니라는걸 명시해줌
# train_csv = pd.rean_csv('./_data/ddareum/train.csv' , index_col=0) // 위에 패치를 안했다면 계속 불러와야함
test_csv = pd.read_csv(path + 'test.csv' , index_col=0)
submission = pd.read_csv(path+ 'submission.csv' , index_col=0)

print(train_csv)
print(train_csv.shape) #(1459,10)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#         dtype='object')
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

# 결측치 처리 1. 제거
print(train_csv.isnull().sum()) # nan값의 갯수를 구한다.
train_csv = train_csv.dropna() # nan값을 전부 빼버린다.
print(train_csv.isnull().sum())
print(train_csv.shape)

x = train_csv.drop(['count'],axis=1) # train값에서 count의 행을 뺀다.
print(x) #Name: count, Length: 1459, dtype: float64
y = train_csv['count']
print(y)
print(y.shape) #(1459,)
print(submission.shape) #(715, 1)


x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    shuffle=True,
                                                    train_size=0.8,
                                                    random_state=40)

print(x_train.shape, x_test.shape)  #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape)  #(1021,) (438,)

# 2.모델구성
model = Sequential()
model.add(Dense(6,input_dim=9)) #  count를 잘랐기 때문에 10개가 아닌 9개
model.add(Dense(45))
model.add(Dense(70))
model.add(Dense(1))

# 3.컴파일,훈련

model.compile(loss='mse',optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train,y_train,epochs=101,batch_size=30,validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)


# 4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)

y_submit = model.predict(test_csv)
submission['count'] = y_submit

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

submission.to_csv( './_data/test/ddarung/' + "submission_0109_01.csv")


'''
[6693.02099609375, 6083.33544921875, 6302.888671875, 5778.00830078125, 5467.37255859375, 6793.96142578125, 4984.14013671875, 5255.66650390625, 4915.8017578125, 4175.34765625, 5889.71728515625, 3998.33154296875, 3709.09326171875, 3583.765625, 3400.820068359375, 3454.81787109375, 3629.834228515625, 5028.6064453125, 3598.289306640625, 3749.431640625]

'''