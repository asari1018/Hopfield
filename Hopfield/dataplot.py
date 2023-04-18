import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def dataplot(train, test, result):
        fig, axes = plt.subplots(len(train),3)
        dim = int(np.sqrt(len(train[0])))

        axes[0][0].set_title('train data')
        axes[0][1].set_title('test data')
        axes[0][2].set_title('result')

        for i in range(len(train)):
            img = (train[i].reshape(dim, dim) + 1) / 2
            axes[i][0].imshow(img, cmap=cm.Greys_r, interpolation='nearest')
            img = (test[i].reshape(dim, dim) + 1) / 2
            axes[i][1].imshow(img, cmap=cm.Greys_r, interpolation='nearest')
            img = (result[i].reshape(dim, dim) + 1) / 2
            axes[i][2].imshow(img, cmap=cm.Greys_r, interpolation='nearest')
            
        plt.show()
        return 