from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1.data
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True,
                                                    train_size=0.7,
                                                    random_state=123)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=10))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=5000,batch_size=1)

# 4.prediction
y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_absolute_error(y_test,y_predict))
print("RMSE :",RMSE(y_test,y_predict))

r2 = r2_score(y_test,y_predict)
print("R2 :",r2) 

'''
RMSE : 6.645735148442619
R2 : 0.4980234699515148

'''