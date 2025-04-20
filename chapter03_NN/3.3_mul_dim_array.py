import numpy as np

A = np.array([1, 2, 3])
print(A)
print(A.ndim)
print(A.shape)

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(B)
print(B.ndim)
print(B.shape)

# 矩阵相乘
print(np.dot(B, A))
print(B @ A)