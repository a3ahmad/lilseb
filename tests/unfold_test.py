import torch

a = torch.tensor([1, 2, 3, 4, 5])
print(a)
print(a.unfold(0, 2, 1))
b = torch.einsum('i,j->ij', a, a.T)
print(b)
print(b.unfold(0, 2, 1).unfold(1, 2, 1))
# Convolve with einsum('ij,klij->kl', kernel, sub_matrices)