from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1.데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
print(x.shape)  # (3,10)
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])
# 실습:넘피아 리스트 슬라이싱// 7:3으로 잘라라!!!
# train과 test를 섞어서 7:3을 만든다. 힌트: 사이킷 
# 2. 모델
model = Sequential()
model.add(Dense(6,input_dim=3))
model.add(Dense(7))
model.add(Dense(2))

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.3,
                                                 shuffle=True,
                                                 random_state=123)

