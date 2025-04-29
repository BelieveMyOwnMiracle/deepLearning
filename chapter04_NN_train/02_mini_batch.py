import numpy as np

from dataset.mnist import load_mnist

import sys,os
sys.path.append(os.pardir)

def get_mini_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = x_train.shape[0]
    batch_size = 10

    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = x_test[batch_mask]

    return x_batch, t_batch