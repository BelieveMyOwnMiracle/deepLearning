import numpy as np
from matplotlib import pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def draw_activate_func():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)

    plt.plot(x, y1,label="step_function")
    plt.plot(x, y2,label="sigmoid", linestyle="--")
    plt.plot(x, y3,label="relu", linestyle="dotted")
    plt.ylim(-0.1, 1.1)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_activate_func()