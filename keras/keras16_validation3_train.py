from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 실습 자르기 (10:3:3)
# train_test_split이용하여 자르기
# 1. 데이터

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.65, 
                                                    shuffle=False
                                                    )

x_test, x_validation, y_test, y_validation = train_test_split(x_test,y_test,
                                                                test_size=0.5, 
                                                                shuffle=True
                                                                )

print(x_train)
print(x_test)
print(x_validation)
print(y_train)
print(y_test)
print(y_validation)