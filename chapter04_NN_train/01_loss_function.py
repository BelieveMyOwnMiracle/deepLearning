import numpy as np

def mean_square_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y,t):   # 当正确解标签的预测概率越小时，输出值反而越大
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


if __name__ == "__main__":

    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

    print(mean_square_error(np.array(y), np.array(t)))
    print(cross_entropy_error(np.array(y), np.array(t)))

    print("=" * 100)
    y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    print(mean_square_error(np.array(y), np.array(t)))
    print(cross_entropy_error(np.array(y), np.array(t)))