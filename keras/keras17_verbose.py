from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print (x.shape , y.shape)   #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,random_state=333,test_size=0.2
)

# 2.모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))   # 행과 열로 구성되어있을때만 가능
model.add(Dense(5, input_shape=(13,))) # 다차원으로 나왔을대 사용 //  행우선열무시
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3.컴파일
import time
model.compile(loss='mse',optimizer='adam')
star = time.time()
model.fit(x_train,y_train,epochs=50,batch_size=1,
          validation_split=0.2,
          verbose=2)    
end = time.time()

# 4.평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

print('걸린시간 : ' , end - star)

# verbose 0 : 아무것도 안보여줌
# verbose 1 : 진행 가정을 보여줌(프로세서 바 생성)
# verbose 2 : 진행 과정을 보여주지 않음(프로세서 바 비 생성)
# verbose 3 : epochs만 보여줌.
# verbose로 딜레이를 줄일수 있다. 양이 많으면 그 만큼 속도가 빨라짐
