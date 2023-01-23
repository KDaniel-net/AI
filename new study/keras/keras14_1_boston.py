# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

print(dataset.feature_names)    # 열의 이름을 출력해서 확인한다.
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True,
                                                    train_size=0.7,
                                                    random_state=20)

print(dataset.DESCR)    # load_boston에 대한 설명

# 2. model
model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=500,batch_size=32)

# 4.prediction
model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE(y_test,y_predcit):
    return np.sqrt(mean_absolute_error(y_test,y_predict))
# 예측값과 실제값 사이의 평균오차값의 제곱근

print("RMSE :",RMSE(y_test,y_predict))

r2 = r2_score(y_test,y_predict)
print("R2 :",r2)


