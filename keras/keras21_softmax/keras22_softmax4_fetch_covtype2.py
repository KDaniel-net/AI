from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

# 1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape,y.shape)              # (581012, 54) (581012,)
print(np.unique(y,return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), 
#  array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

print(type(y))  # <class 'numpy.ndarray'>

print(y.shape)          # (581012,)
y = y.reshape(581012, 1)
print(y.shape)          # (581012, 1)
print(y)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder ()
# ohe.fit(y)
# y = ohe.fit_transform(y)
y = ohe.fit_transform(y)
print(y[:15])
print(type(y))
y = y.toarray()



print(np.unique(y,return_counts=True))
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                                 shuffle=True,
                                                 random_state=30,
                                                 test_size=0.2,
                                                 stratify=y)

# 2.모델구성
model = Sequential()
model.add(Dense(5,activation='relu', input_shape=(54,)))
model.add(Dense(4,activation='sigmoid'))                   
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='linear'))
model.add(Dense(7,activation='softmax'))       # 다중 출력 // y의class의 갯수 // 다중 출력일때는 activation의 함수는 softmax이다.

# 3.컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                             mode='min',        # 최소값을 찾아준다. auto,max가 더 있음
                             patience=20, 
                             restore_best_weights=True,
                             verbose=1)

hist = model.fit(x_train,y_train, epochs=1, batch_size=9000,
                                        validation_split=0.2,
                                        callbacks=earlyStopping,
                                        verbose=1)


# 4.평가


from sklearn.metrics import accuracy_score

loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ',loss,'accuracy : ',accuracy)

y_predict = np.argmax(model.predict(x_test), axis=1)        # 예측했던 값 y_predict = model.predict(x_test)를 넣어줌
print('y_predict(예측값) :', y_predict[:20])

y_test = np.argmax(y_test, axis=1)              # y_test를 원핫인코딩 해줬던 값을 다시 원래대로 돌려주는것          
print('y_test(원래값) :',y_test[:20])

acc = accuracy_score(y_test, y_predict)         # 그냥 구하면 정수와 실수이기 때문에 구할수가 없다.
print('acc :' ,acc)


