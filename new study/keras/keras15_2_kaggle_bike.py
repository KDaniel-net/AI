from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
path = './_data/bike/'
path2 = './ns/bike/'
train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',index_col=0)

print(train_csv)
print(train_csv.shape)
print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

##### 결측치 제거 #####
# print(train_csv.isnull().sum())
# train_csv = train_csv.dropna()
# print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)    # [10886 rows x 8 columns] 
y = train_csv['count']
print(y)
print(y.shape)  # (10886,)
print(test_csv) #[6493 rows x 8 columns]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True,
                                                    random_state=20,
                                                    train_size=0.7)
print(x_train.shape, x_test.shape)  # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=8,activation='relu'))
# relu = 입력이 양수인 경우 출력은 입력과 같고, 입력이 음수인 경우 출력은 0입니다.
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1,activation='linear'))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae','acc'])
model.fit(x_train,y_train,epochs=400,batch_size=32)

# 4. prediction
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (6493, 1)
print(submission_csv.shape) # (6493, 1)

submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path2 + 'submission_0124.csv')
print ('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2)
print("loss :",loss)
