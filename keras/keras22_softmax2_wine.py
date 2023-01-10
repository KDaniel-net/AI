import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1.데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR) # pandas: .describe() / .info()
# print(datasets.feature_names) # pandas: .columns
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)         # (178, 13) (178,)
print(np.unique(y))             # [0 1 2]
print(np.unique(y,return_counts=True))            
# array([0, 1, 2]), array([59, 71, 48]
# 분류 데이터임을 알수 있다. 

y = to_categorical(y)
# y에 해당하는 카탈로그르 만들어줌
# np.unique(y)으로 확인한 0,1,2에 해당하는 열을 만들어줌


x_train,x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True,                   # False의 문제점 라벨값들이 동일하게 되기때문에 성능이 좋지 않다. (한쪽으로 몰림현상 발생)
    random_state=40,
    test_size=0.2,
    stratify=y)                         # 같은 비율로 떨어지게 만들어 준다. 예) 첫번째 : [1 1 2 0 2 1 2 1 0 0 2 0 0 1 2]   두번째 : [0 0 2 0 2 1 0 1 0 2 2 2 1 1 1]

# 2.모델구성
model = Sequential()
model.add(Dense(5, activation='relu' , input_shape=(13,)))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(13))
model.add(Dense(3,activation='softmax'))
# 다중 분류에서는 softmax만 사용한다. 확률의 총합은 1, 대게 output층에서 사용한다. 

# 3.컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss',                # 무슨 값을 기준으로 할것인지
                             mode='max',                        # 최소값을 찾아준다. auto,max가 있음
                             patience=10,                        # 몇번을 참아줄지/ 예를들어 5면 출력값에 뒤에서 5번째 앞에 값이 제일 크다.(함수에 따라 달라짐)
                             restore_best_weights=True,
                             verbose=1)

hist = model.fit(x_train,y_train,epochs=100,batch_size=1,   # 맨 앞에 붙은 hist는 결과 값. 즉, 내용을 history에 담겠다는 의미
          validation_split=0.2, # 0.2만큼 잘라서 validation값으로 사용한다.
          callbacks=[earlyStopping],
          verbose=2)
# 4.평가
loss, accuracy = model.evaluate(x_test,y_test)
print('loss :',loss,'accuracy: ',accuracy)

y_predict = np.argmax(model.predict(x_test), axis=1)
print('y_predict(예측값) :', y_predict)

y_test = np.argmax(y_test, axis=1)
print('y_test(원래값) :',y_test)

acc = accuracy_score(y_test, y_predict) 
print('acc :',acc)

'''
acc : 0.4166666666666667
'''