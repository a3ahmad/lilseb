import torch

# This is useless until einsum supports sparse tensors
def to_sparse(dense):
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1], indices[2]]
    return torch.sparse.FloatTensor(indices, values, dense.size())

