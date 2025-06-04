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
    t_batch = t_train[batch_mask]

    return x_batch, t_batch

def cross_entry_loss_batch_1(y, t): # t是 one-hot形式的标签
    if y.ndim == 1:
        y = y.reshpae(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size #正则化

def cross_entry_loss_batch_2(y, t):
    if y.ndim == 1:
        y = y.reshpae(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size #正则化