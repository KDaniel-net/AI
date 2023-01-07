from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])
# x,y또는 그외의 이름을 지정하고 거기에 배열로 된 1,2,3,4,5를 넣겠다는 의미

# 2.모델구성
model = Sequential()
model.add(Dense(6, input_dim=1))
# 최초의 입력할 값을 넣음! 그 후 출력값의 갯수 // input,output
# input layer(입력층)
model.add(Dense(2))
# 처음에 출력된 값을 다시 사용하기 때문에 input값을 따로 넣지 않아도 됨!
# 마지막 모델링은 출력되는 값을 나타내는것
# hidden layer(숨은층)
model.add(Dense(1))
# output layer(출력층)

# 3.컴파일
model.compile(loss='mae',optimizer='adam')
# mae : 예측값과 결과값의 평균을 나타낸 값

model.fit(x,y,epochs=100)
# epochs는 몇번 돌릴지!! 즉, 몇번 계산을 할지에 대하여 정하는 문구이다.

# 4.평가
result = model.predict([6])
print('6의 예측값 : ' , result)
