import numpy as np

a = np.ones((8,3))
b = np.ones(3)
print(a)
print(b)
c = np.matmul(np.transpose(a), b)
print(c)