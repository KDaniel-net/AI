from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.정제된 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2.모델구성
model = Sequential()
model.add(Dense(20,input_dim=1))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(1))

# 3.컴파일
model.compile(loss='mae' , optimizer='adam')
model.fit(x,y,epochs=10,batch_size=1)
# batch_size는 테이터를 어떻게 잘라서 계산할지를 이야기한다. 
# Default 값은 32이다.

# 4.평가,예측
result=(model.predict([6]))
print("6의 예측값 : " , result)

'''
6의 예측값 :  [[5.7878194]]
'''