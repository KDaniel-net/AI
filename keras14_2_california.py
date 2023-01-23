# [실습]
# R2 0.55~0.6 이상

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
dataset = fetch_california_housing()
print(dataset)

x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (20640, 8) (20640,)

print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(dataset.DESCR) 데이터의 구성을 확인한다. // 굳이 안봐도 상관없음

x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.7,
                                                     shuffle=True,
                                                     random_state=20)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=8))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=500,batch_size=50)

# 4.prediction
# loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_absolute_error(y_test,y_predict))
print("RMSE :",RMSE(y_test,y_predict))

r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

'''
RMSE : 0.7665846739739273
R2 : 0.5483581897228951
'''