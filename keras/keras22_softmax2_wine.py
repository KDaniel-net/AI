import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (178, 13) (178,)
print(y)
print(np.unique(y))             # [0 1 2]
print(np.unique(y,return_counts=True))            
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder

y = ohe.fit_transform(y)
print(y)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True,                   # False의 문제점 라벨값들이 동일하게 되기때문에 성능이 좋지 않다. (한쪽으로 몰림현상 발생)
    random_state=40,
    test_size=0.2,
    stratify=y                          # 같은 비율로 떨어지게 만들어 준다. 예) 첫번째 : [1 1 2 0 2 1 2 1 0 0 2 0 0 1 2]   두번째 : [0 0 2 0 2 1 0 1 0 2 2 2 1 1 1]
)


print(y)

# 2.모델구성
model = Sequential()
model.add(Dense(5, activation='relu' , input_shape=(13,)))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3,activation='softmax'))

# 3.컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=150,batch_size=1,
          validation_split=0.2,
          verbose=2)

from tensorflow.keras.callbacks import EarlyStopping

earlyStoping = EarlyStopping(monitor='val_loss',                # 무슨 값을 기준으로 할것인지
                             mode='max',                        # 최소값을 찾아준다. auto,max가 더 있음
                             patience=5,                        # 몇번을 참아줄지
                             restore_best_weights=True,
                             verbose=1)

# 4.평가
loss, accuracy = model.evaluate(x_test,y_test)

print('========================')
print(hist) #<keras.callbacks.History object at 0x0000016F21A30880>
print('========================')
print(hist.history) # hist안에 있는 리스트를 보여줌. history에는 loss값 val_loss값의 형태가 들어감
print('========================')
print(hist.history['loss'])    # hist 안에 있는 loss값만을 보고 싶다.
print('========================')
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict) 

print('loss : ', loss)
print('accuracy : ', accuracy)
