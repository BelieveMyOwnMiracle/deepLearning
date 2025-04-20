import numpy as np

def sigmod(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=int)

def identity_function(x):
    return x

def softmax(a):
    max_a = np.max(a)
    exp_a = np.exp(a-max_a)
    return exp_a / np.sum(exp_a)

def relu(x):
    return np.maximum(x, 0)