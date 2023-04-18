import numpy as np
import copy

def predict(data, J, threshold):
    result = copy.deepcopy(data)

    for i,s in enumerate(result):
        #データの最初のエネルギー
        E = -s.dot(J).dot(s) + np.sum(s * threshold)
        #エネルギーが変化しなくなるまでor100回終わるまで繰り返す
        for l in range(100):
            #s(t+1) = sgn(sigma(Js)-threshold)
            s = np.sign(J.dot(s) - threshold)
            #学習後のエネルギー
            E_new = -s.dot(J).dot(s) + np.sum(s * threshold)
            #エネルギーが変化しなくなったら抜ける
            if(abs(E - E_new) < 0.001) : break
            #最新のエネルギーを更新
            E = E_new
        #収束した値を結果とする
        result[i] = s

    return result


