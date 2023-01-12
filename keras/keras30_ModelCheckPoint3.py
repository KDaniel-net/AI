from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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


# 3.컴파일
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

es = EarlyStopping(monitor='val_loss',
                              mode='min',               # val_loss는 낮을수록 좋다
                              patience=20,
                            #   restore_best_weights=False,             #기본값이 False
                              verbose=1)

mcp = ModelCheckpoint(monitor='val_loss', 
                      mode='auto', 
                      verbos=1, 
                      save_best_only=True,          # 가장 좋은 지점만 저장해라 
                      filepath= path+ 'MCP/keras30_model_ModelCheckPoint3.hdf5')

model.fit(x_train,y_train,epochs=5000,batch_size=1,validation_split=0.2,callbacks=[es, mcp],verbose=1)


# model.save( path + 'keras29_3_svae_model.h5')
# # model.save( './_save/keras29_1_svae_model.h5')
# #  0.711610702423874

model.save(path+'keras30_ModelCheckPoint3_save_model.h5')

# model = load_model(path+ 'MCP/keras30_model_ModelCheckPoint1.hdf5')

# 4.평가,예측
print('================== 1. 기본 출력 ==================')
mse = model.evaluate(x_test,y_test)
print ('mse :' , mse )

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print(' r2스코어 : ' , r2)

print('================== 2.load_model 기본 출력 ==================')
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse = model2.evaluate(x_test,y_test)
print ('mse :' , mse )

y_predict = model2.predict(x_test)

r2 = r2_score(y_test, y_predict)
print(' r2스코어 : ' , r2)

print('================== 3. ModelCheckPoint 기본 출력 ==================')
model3 = load_model(path + 'MCP/keras30_model_ModelCheckPoint3.hdf5')
mse = model3.evaluate(x_test,y_test)
print ('mse :' , mse )

y_predict = model3.predict(x_test)

r2 = r2_score(y_test, y_predict)
print(' r2스코어 : ' , r2)
'''
MCP : 0.9232766676173156

'''