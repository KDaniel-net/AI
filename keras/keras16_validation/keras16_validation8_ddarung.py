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
model.fit(x_train,y_train,epochs=101,batch_size=30,validation_split=0.2)


# 4. 평가,예측

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)

# 제출
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) 

submission['count'] = y_submit
print(submission)

# submission.to_csv(path + "submission_0108_03.csv")

r2 = r2_score(y_test, y_predict)
print(" R2 : ",r2)
print("RMSE :", rmse)

end = time.time()
print('걸린시간 : ' , end - start)

'''
R2 :  0.5844738755666032
RMSE : 50.36306190776473

R2 :  0.5727516171661691
RMSE : 51.06850837033577

 R2 :  0.5871023028490101
RMSE : 50.20352242307889
걸린시간 :  9.37826132774353

 R2 :  0.4969144067958947
RMSE : 57.13883525958819
걸린시간 :  10.274389505386353

 R2 :  0.4938826939607186
RMSE : 57.31074273114707
걸린시간 :  13.888822555541992
'''