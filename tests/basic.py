import numpy as np

from lilseb.algebra import *
from lilseb.pytorch import *

A = np.array([[0,0,0,0,-1],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[-1,0,0,0,0]])
M = Metric(A)
e1 = np.zeros(M.basis_dim())
e1[0b1] = 1.0
e2 = np.zeros(M.basis_dim())
e2[0b10] = 1.0
e3 = np.zeros(M.basis_dim())
e3[0b100] = 1.0
e4 = np.zeros(M.basis_dim())
e4[0b1000] = 1.0
e5 = np.zeros(M.basis_dim())
e5[0b10000] = 1.0
r = geometric_product(e1, e5, M)
indices = np.nonzero(r)[0]
print(indices)
print("{0:b}".format(indices[-1]))
print(r[indices])

print("---")

r = np.einsum('ijk,i,j->k', M.get_geometric_product(), e1 , e5)
indices = np.nonzero(r)[0]
print(indices)
print("{0:b}".format(indices[-1]))
print(r[indices])

print("---")

a = np.random.rand(3, 8, 32)
b = np.random.rand(8, 16, 32)
result = np.einsum('ijk,bpi,poj->bok', M.get_geometric_product(), a, b)
print(result.shape)

print("---")

A = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,-1],[0,0,0,-1,0]])
M = Metric(A)
e1 = np.zeros(M.basis_dim())
e1[0b1] = 1.0
e2 = np.zeros(M.basis_dim())
e2[0b10] = 1.0
e3 = np.zeros(M.basis_dim())
e3[0b100] = 1.0
e4 = np.zeros(M.basis_dim())
e4[0b1000] = 1.0
e5 = np.zeros(M.basis_dim())
e5[0b10000] = 1.0

def print_indices(indices, r):
    if len(indices) == 0:
        return "     0"
    label = ""
    for idx in indices:
        if idx == 0:
            name = str(r[idx])
        else:
            name = format(idx, "05b")

        if label == "":
            if r[idx] < 0.0:
                label = "-" + name if idx > 0 else name
            else:
                label = " " + name
        else:
            if r[idx] < 0.0:
                label = label + " - " + name
            else:
                label = label + " + " + name
        
    return label

e = [e1, e2, e3, e4, e5]
for i in range(5):
    for j in range(5):
        r = np.einsum('ijk,i,j->k', M.get_geometric_product(), e[i] , e[j])
        indices = np.nonzero(r)[0]
        print (print_indices(indices, r) + " ", end='')
    print("")
