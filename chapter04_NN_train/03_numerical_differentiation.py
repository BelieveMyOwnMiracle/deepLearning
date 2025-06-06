
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x

import matplotlib.pylab as plt
import numpy as np

x = np.arange(0.0, 20.0, 0.1) # 以 0.1 为单位，从 0 到 20 的数组 x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

