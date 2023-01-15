from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
y = np.array(range(10))                 # (10, )

# [검색] train과 test를 섞어서 7:3으로 만들기!
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.3,          
                                                    # test의 비율을 어떻게 할지를 말한다. 반대로 train을 적을수도 있다.
                                                    shuffle=False,
                                                    # 데이터를 분할하기 전에 미리 섞을지에 대한 명령어다. 기본값은 default는 Treu
                                                    random_state=100)
                                                    # 같은 상태로 지정하게 해준다. 


# 2. model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

# 3.compile
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=1)

# 4. prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

result = model.predict([11])
print('[11]의 결과 : ', result)

''' loss : 0.04642804339528084
[11]의 결과 :  [[9.932182]] '''