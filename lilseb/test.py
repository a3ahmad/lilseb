import numpy as np
from algebra import *
A = np.array([[0,0,0,0,-1],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[-1,0,0,0,0]])
M = Metric(A)
v1 = np.zeros(32)
v1[0b0] = 1.0
v2 = np.zeros(32)
v2[0b1] = 1.0
v3 = np.zeros(32)
v3[0b10] = 1.0
v5 = np.zeros(32)
v5[0b10000] = 1.0
r = geometricProduct(v2, v5, M)
indices = np.nonzero(r)[0]
print(indices)
print("{0:b}".format(indices[-1]))
print(r[indices])
