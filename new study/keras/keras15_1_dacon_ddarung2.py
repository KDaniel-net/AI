from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 1.data
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv' ,index_col=0)
test_csv = pd.read_csv(path + 'test.csv' ,index_col=0)
submission = pd.read_csv(path + 'submission.csv' ,index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10)
print(train_csv.columns)

##### 결측치 제거 #####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)  # (1328, 10)

print(train_csv.info())
# null(결측치)를 확인할수 있다.
print(test_csv.info())
print(train_csv.describe())
# 각열의 갯수, 평균, 표준 편차 등을 알려준다.

x = train_csv.drop(['count'], axis=1)
# train_csv에 count라는 떨구라는 의미. 
# "axis=0"은 행을 의미하며, "axis=1"은 열을 의미
print(x)    # [1459 rows x 9 columns] // 열이 떨궈진거를 확인할수 있다.
y = train_csv['count']
print(y.shape) # (1459,0)
print(test_csv)

x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     shuffle=True,
                                                     train_size=0.7,
                                                     random_state=20)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=9))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=10,batch_size=32)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print("loss :",loss)

y_predict = model.predict(x_test)
print("y_predict :", y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :",RMSE(y_test,y_predict))

r2 = r2_score(y_test,y_predict)
print("r2 :", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)
print(submission.shape)

submission['count'] = y_submit
print(submission)

submission.to_csv( path + ' submission_0124.csv ')

# panda는 데이터 분석시 사용하기 좋은 API이다. : CSV read할 때에도 쓰임 : 데이터분석쪽의 scikit-learn 같은 의미이다.
# pandas가 좋은 이유가 print하면 컬럼명과 행열수 바로 알려줌.
