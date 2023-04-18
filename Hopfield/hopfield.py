import numpy as np
from fit import fit
from predict import predict
#from dataplot import dataplot
from datanoise import datanoise
from matplotlib import pyplot as plt

#閾値の設定
threshold = 0
#記憶する画像の辺の長さ
size = 50
#ノイズを加えたデータの割合
noise_param = 0.7
size *= size
a_data = []
m_data = []

for p in range(1,250):
    mem_num = p
    alpha = mem_num/size
    a_data.append(alpha)
    train_data = []

    #記憶するデータ
    for i in range(mem_num):
        train_data.append(np.random.randint(0, 2, size))
        for j in range(size):
            if train_data[i][j] == 0:
                train_data[i][j] = -1

    #ノイズを加えたデータの作成
    test_data = datanoise(train_data,noise_param)

    result = predict(test_data, fit(train_data), threshold)
    
    m = 0
    for i in range(p):
        m += (train_data[i].dot(result[i]))/size
    m /= p
    m_data.append(m)
    print(alpha, m)
    
    '''
    print("train is ", train_data)
    print("test is ", test_data)
    print("result is ", result)
    dataplot(train_data, test_data, result)
    '''

plt.plot(a_data, m_data)
plt.show()