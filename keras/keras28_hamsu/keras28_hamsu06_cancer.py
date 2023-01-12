from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape,y.shape)          #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,random_state=333,test_size=0.2

)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# 2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear',input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))              # 이진은 나올때 0 ~ 1 이 나와야 함으로 sigmoid

# 2.모델구성(함수형)
x = Input(shape=(30,))
d1 = Dense(40)(x)
d2 = Dense(30)(d1)
d3 = Dense(10)(d2)
y = Dense(1)(d3)
model = Model(inputs = x, outputs = y)
model.summary()

# 3.컴파일
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])             # 이진 loss값은 무조건 binary_crossentroy

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                             mode='min',        # 최소값을 찾아준다. auto,max가 더 있음
                             patience=20, 
                             restore_best_weights=True,
                             verbose=1)
model.fit(x_train,y_train,epochs=10000,batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

# 4.평가,예측
loss,accuracy = model.evaluate(x_test,y_test)
print('loss :',loss)
print('accuracy :',accuracy)
import numpy as np
y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                # 보기 쉽게 하기 위하여  
y_predict = np.where(y_predict > 0.5, 1 , 0)
# 정수값으로 변환해준다. // 반올림
    
print(y_predict[:10])              
print(y_test[:10])

from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test,y_predict)
# print("accuracy_score :" , acc)
'''
loss : 0.15912650525569916
accuracy : 0.9561403393745422

minmaxscaler
oss : 0.14500704407691956
accuracy : 0.9649122953414917

standardsclaer
loss : 0.15236012637615204
accuracy : 0.9649122953414917
'''