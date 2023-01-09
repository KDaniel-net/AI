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
          verbose=4)    # verbose 과정을 보여줄지, 0 = 안보여줌(딜레이 걸리는 시간이 줄어든다.) 1 = 보여줌(진행 형식을 보여줌) 2 = 진행 과정을 보여주지 않음(프로그레스 바 삭제) 3= 에포만 보여줌
end = time.time()

# 4.평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

print('걸린시간 : ' , end - star)
