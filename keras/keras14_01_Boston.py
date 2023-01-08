from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
dataset = load_boston()

print(dataset.feature_names) #속성의 이름값을 가져옴

x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(506, 13)
print(y)
print(y.shape)  #(506,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=1)
print(dataset.DESCR)

# 2.모델구성
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(6))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print ('loss :' , loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE : ' , RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print(' r2 : ' , r2)
