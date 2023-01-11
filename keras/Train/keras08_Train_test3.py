from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # (10, )
y = np.array(range(10))  # (10, )

# 실습:넘피아 리스트 슬라이싱// 7:3으로 잘라라!!!
# train과 test를 섞어서 7:3을 만든다. 힌트: 사이킷 런

# 2. 모델구성
model = Sequential()
model.add(Dense(8,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.7,
                                                    shuffle=True,
                                                    random_state=34
                                                    )

# test_size의 default값 0.25 // 3:7로 자름
# shuffle의 default값은 ture이며, spilt하기 전에 shuffle의 결정하는것
print(x_train)
print(x_test)
print(y_train)
print(y_test)

'''
[6 7 2]
[ 1 10  9  3  8  4  5]
[5 6 1]
[0 9 8 2 7 3 4]
'''