from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

# 1.데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

# print('최소값 : ',np.min(x))                # x의 최저값을 본다.
# print('최대값 : ',np.max(x))                # x의 최대값을 본다.

print(x)
print(type(x))                  # <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=1)
print(dataset.DESCR)

scaler = MinMaxScaler()    
# scaler = StandardScaler()
scaler.fit(x_train)                   # 범위 만큼의 가중치를 생성해준다.
# x_trian = scaler.fit.transform(x_train)         #위의 2둘과 같은 내용이다.
x_train = scaler.transform(x_train)         # x에 변환해서 넣어준다. 
x_test = scaler.transform(x_test)         # x에 변환해서 넣어준다. 

# 2.모델구성 (함수형)
path = './_save/'                   # study그룹에서 작업을 진행할시
# path = '../_save/'                # keras그룹에서 작업을 진행할시 
# path = 'c:/_study/_save/'         # 절대값으로 설정

# model.save( path + 'keras29_1_svae_model.h5')
# model.save( './_save/keras29_1_svae_model.h5')


model = load_model( path + 'keras29_1_svae_model.h5')
model.summary()



# 3.컴파일
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=10,batch_size=5,validation_split=0.2)

# 4.평가,예측
mse, mae = model.evaluate(x_test,y_test)
print ('mse :' , mse , 'mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE : ' , RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print(' r2 : ' , r2)

