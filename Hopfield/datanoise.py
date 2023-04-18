import copy
import numpy as np

def datanoise(data,noise_param):

    noised = copy.deepcopy(data)
    noise_size = int(len(data[0])*noise_param)

    for d in noised:
        noise_id = np.random.randint(0, len(data[0]), noise_size)
        for i in noise_id :
            d[i] *= -1

    return noised
