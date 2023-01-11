from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1.데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)                   # 범위 만큼의 가중치를 생성해준다.
x = scaler.transform(x)         # x에 변환해서 넣어준다. 

print('최소값 : ',np.min(x))                # x의 최저값을 본다.
print('최대값 : ',np.max(x))                # x의 최대값을 본다.

print(x)
print(type(x))                  # <class 'numpy.ndarray'>

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
model.fit(x_train,y_train,epochs=5,batch_size=1,validation_split=0.2)

# 4.평가,예측
mse, mae = model.evaluate(x_test,y_test)
print ('mse :' , mse , 'mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE : ' , RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print(' r2 : ' , r2)


'''
loss : [39.0808219909668, 4.607749938964844]
RMSE :  6.251465677035281
 r2 :  0.6054772327106046

변환후
mse : 66.30453491210938 mae :  5.570650577545166
RMSE :  8.142759439983319
 r2 :  0.3306526042992167
 
'''