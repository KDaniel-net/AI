from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(442, 10)
print(y)
print(y.shape)  #(442,)

print(dataset.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(dataset.DESCR)
# 사이킥런의 데이터 요약

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=123
                                                    )
print(x_train.shape)    #(331, 10)

# 2.모델구성
model = Sequential()
model.add(Dense(80, input_dim=10))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=101, batch_size=9,validation_split=0.3)

# 4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict))
#  mse값에 루트로 값을 출력하는것

r2 = r2_score(y_test, y_predict)
print(" R2 : ", r2)

'''
loss : [3209.5068359375, 47.17751693725586]
RMSE : 56.65251038644996
 R2 :  0.4601762411553115
 
 loss : [3135.6953125, 44.925758361816406]
RMSE : 55.99727657006973
 R2 :  0.4727008283212688
 
 loss : [3007.12939453125, 44.85442352294922]
RMSE : 54.83730135044699
 R2 :  0.4943204087442602
 
 loss : [2980.453125, 44.887142181396484]
RMSE : 54.59352486158221
 R2 :  0.49880636230818653
'''