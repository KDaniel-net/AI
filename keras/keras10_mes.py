from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


# 1.데이터
x = np.array(range(1, 21))
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13,
             8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     test_size=0.7,
                                                     shuffle=True,
                                                     random_state=123)
# 2.모델
model = Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=1)

# 4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

'''
mae : loss : 4.1562347412109375
mse : loss : 181.2904052734375
'''