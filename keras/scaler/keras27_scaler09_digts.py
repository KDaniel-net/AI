import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_digits()

x = datasets.data
y = datasets['target']

print(x.shape,y.shape)     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
#  array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

y = to_categorical(y)
print(type(y))      # y의 클래스 타입을 알수 있다. <class 'numpy.ndarray'>

x_train,x_test,y_train,y_test = train_test_split(
    x, y, shuffle=True,
    random_state=12,
    test_size=0.2,
    stratify=y
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# 2.모델구성
model = Sequential()
model.add(Dense(5,activation='relu',input_shape=(64,)))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(10,activation='softmax'))

# 3.컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss',
                             mode='max',
                             patience=5,
                             restore_best_weights=True,
                             verbose=1)

hist = model.fit(x_train,y_train, epochs=100, 
                                batch_size=3,
                                validation_split=0.2,
                                callbacks=[earlyStopping],
                                verbose=2)





# 4.평가

loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ',loss,'accuracy : ',accuracy)

y_predict = np.argmax(model.predict(x_test), axis=1)        # 예측했던 값 y_predict = model.predict(x_test)를 넣어줌
print('y_predict(예측값) :', y_predict[:10])

y_test = np.argmax(y_test, axis=1)              # y_test를 원핫인코딩 해줬던 값을 다시 원래대로 돌려주는것          
print('y_test(원래값) :',y_test[:10])

acc = accuracy_score(y_test, y_predict)         # 그냥 구하면 정수와 실수이기 때문에 구할수가 없다.
print('acc :' ,acc)

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[4])
# plt.show()


'''
minmax
acc : 0.33611111111111114

standard
acc : 0.4

'''