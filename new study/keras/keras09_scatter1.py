from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#1. data
x = np.array(range(1,21)) # (20, )
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) # (20, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=213)

# 2. model
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4.prediction
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict,color='red')
plt.show()