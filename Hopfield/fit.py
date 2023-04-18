import numpy as np

def fit(data_list):

    #学習するデータの個数
    p = len(data_list)
    #学習するデータの長さ
    N = len(data_list[0])
    
    #学習値Jの初期化
    J = np.zeros((N, N))

    #Jに全学習データを重ね合わせる
    for myu in range(p):
        J += np.outer(data_list[myu], data_list[myu])

    #Jの対角成分は0
    for i in range(N):
        J[i,i] = 0

    #パターンをsumしたJを1/Nする
    J /= N

    #学習した行列Jを返す
    return J
