from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # (10, )
            # 0  1  2  3  4  5  6  7  8  9 배열의 숫자를 생각할때/ 0부터
y = np.array(range(10))                         # (10, )

x_train = x[0:7]
# 0번 부터 7-1까지 출력
x_test = x[7:]
y_train = y[0:7]
y_test = y[7:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)
