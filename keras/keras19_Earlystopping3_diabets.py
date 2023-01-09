from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print (x.shape , y.shape)   #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,random_state=333,test_size=0.2
)

# 2.모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))   # 행과 열로 구성되어있을때만 가능
model.add(Dense(5, input_shape=(10,))) # 다차원으로 나왔을대 사용 //  행우선열무시
model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(20000))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mse',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train,y_train,epochs=300,batch_size=1,            
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)

# 4.평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

print('========================')
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
plt.title('diavetes loss')
plt.legend()                        # 범주 만들어줌
# plt.legend(loc='upper left')        # 범주의 위치
plt.show()

'''
[2656.503173828125, 2405.1259765625, 2217.875732421875, 2287.80322265625, 3027.563720703125, 2226.794677734375, 2367.7314453125, 2320.011962890625]


'''