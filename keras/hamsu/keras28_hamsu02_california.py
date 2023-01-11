from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model

# 1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print (x.shape , y.shape)   #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,random_state=333,test_size=0.2
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

''' # 2.모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))   # 행과 열로 구성되어있을때만 가능
model.add(Dense(5, input_shape=(8,))) # 다차원으로 나왔을대 사용 //  행우선열무시
model.add(Dense(40))
model.add(Dense(3))
model.add(Dense(200))
model.add(Dense(1)) '''

# 2.모델구성(함수형)
x = Input(shape=(8,))
D1 = Dense(50)(x)
D2 = Dense(40)(D1)
D3 = Dense(30)(D2)
D4 = Dense(20)(D3)
y = Dense(1)(D4)
model = Model(inputs=x, outputs=y)
model.summary()

# 3.컴파일
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train,y_train,epochs=10,batch_size=1000,            
          validation_split=0.2,
          verbose=3)

# 4.평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

''' print('========================')
print(hist) #<keras.callbacks.History object at 0x0000016F21A30880>
print('========================')
print(hist.history) # hist안에 있는 리스트를 보여줌. history에는 loss값 val_loss값의 형태가 들어감
print('========================')
print(hist.history['loss'])    # hist 안에 있는 loss값만을 보고 싶다.
print('========================')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c='red', 
         marker='.', label ='loss')      # 점 형식으로 이름으 loss
plt.plot(hist.history['val_loss'],c='blue', 
         marker='.', label = 'val_loss')
plt.grid()                          # 그래프에 격자 추가
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('califrnia loss')
plt.legend()                        # 범주 만들어줌
# plt.legend(loc='upper left')        # 범주의 위치
plt.show() '''


'''
MinMaxScaler
loss :  0.6326467990875244

StandardScaler
loss :  2.498596668243408

함수형 출력
loss :  0.6458807587623596
'''