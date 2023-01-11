from sklearn.datasets import load_iris , load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
# 1.데이터 
datasets = load_iris()
# datasets = load_boston()

# print(datasets.DESCR)                 # 판다스 .describe() /.info()
# print(datasets.feature_names)         # 데이터셋의 이름을 출력해줌 // 판다스 .columns

x = datasets.data
y = datasets['target']                  # .target 도 가능                   
print(x)
print(y)
      
# print(x.shape,y.shape)                # (150, 4) (150,)

# y = tf.one_hot(y,3)
y = to_categorical(y)   # 원핫인코딩

x_train,x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True,                   # False의 문제점 라벨값들이 동일하게 되기때문에 성능이 좋지 않다. (한쪽으로 몰림현상 발생)
    random_state=333,
    test_size=0.2,
    stratify=y                          # 같은 비율로 떨어지게 만들어 준다. 예) 첫번째 : [1 1 2 0 2 1 2 1 0 0 2 0 0 1 2]   두번째 : [0 0 2 0 2 1 0 1 0 2 2 2 1 1 1]
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# print(y_train)
# print(y_test)

print(y)

# 2.모델구성
model = Sequential()
model.add(Dense(5,activation='relu', input_shape=(4,)))
model.add(Dense(4,activation='sigmoid'))                   
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='linear'))
model.add(Dense(3,activation='softmax'))       # 다중 출력 // y의class의 갯수 // 다중 출력일때는 activation의 함수는 softmax이다.

# 3.컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

# 4.평가
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_prdict = model.predict(x_test[:5])
# print(y_prdict)

from sklearn.metrics import accuracy_score
import numpy as np

y_predict = model.predict(x_test)
print(y_predict)

y_predict = np.argmax(y_predict, axis=1)        # 예측했던 값 y_predict = model.predict(x_test)를 넣어줌
print(y_predict)


y_test = np.argmax(y_test, axis=1)              # y_test를 원핫인코딩 해줬던 값을 다시 원래대로 돌려주는것          
print(y_test)

acc = accuracy_score(y_test, y_predict)         # 그냥 구하면 정수와 실수이기 때문에 구할수가 없다.
print(acc)
