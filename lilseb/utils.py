import numpy as np
import torch

def bit_count(x):
    x = int(x)
    return sum(b == '1' for b in bin(x))
    #3.10 return x.bit_count()

def is_symmetric(M, eps=1e-6):
    return np.all(np.abs(M - M.T) < eps)

def is_diagonal(M, eps=1e-6):
    return np.all(np.abs(M - np.diag(np.diagonal(M))) < eps)

def to_sparse(dense):
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1], indices[2]]
    torch.sparse.FloatTensor(indices, values, dense.size())
