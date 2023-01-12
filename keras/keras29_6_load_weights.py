from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

path = './_save/'                   # study그룹에서 작업을 진행할시
# path = '../_save/'                # keras그룹에서 작업을 진행할시 
# path = 'c:/_study/_save/'         # 절대값으로 설정

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
input1 = Input(shape=(13,))
dense1 =Dense(50, activation='relu')(input1)
dense2 =Dense(40, activation='sigmoid')(dense1)
dense3 =Dense(30, activation='relu')(dense2)
dense4 =Dense(20, activation='linear')(dense3)
output1 = Dense(1,activation='linear')(dense4)
model =Model(inputs=input1, outputs=output1)
model.summary()
# Total params: 4,611

# model.save_weights( path + 'keras29_5_svae_weights_1.h5')
# model.save( './_save/keras29_1_svae_model.h5')
#  0.711610702423874

''' model.load_weights(path + 'keras29_5_svae_weights_1.h5')
# 사용하려면 모델이 정의가 되어 있어야 한다.
# 훈련이 되어있지 않은 데이터가 저장이 되어있기 model.fit을 꼭 지정하고 사용하여야 한다. '''

# 3.컴파일

earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              restore_best_weights=True,
                              verbose=1)

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=10,batch_size=5,validation_split=0.2)

model.load_weights(path + 'keras29_5_svae_weights_2.h5')
# model.save_weights( path + 'keras29_5_svae_weights_2.h5')
# model.save( './_save/keras29_1_svae_model.h5')
#  0.711610702423874


# 4.평가,예측
mse, mae = model.evaluate(x_test,y_test)
print ('mse :' , mse , 'mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE : ' , RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print(' r2 : ' , r2)

