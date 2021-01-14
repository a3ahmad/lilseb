import numpy as np

from .utils import bit_count, is_symmetric

def canonical_reordering_sign_euclidean(a_idx, b_idx):
    a = a_idx >> 1

    accum = 0
    while a != 0:
        accum = accum + bit_count(a & b_idx)
        a = a >> 1

    return 1.0 if (accum & 1) == 0 else -1.0

def canonical_reordering_sign(a_idx, b_idx, metric):
    bitmap = a_idx & b_idx
    output_sign = canonical_reordering_sign_euclidean(a_idx, b_idx)

    if metric != None:
        i = 0
        while bitmap != 0:
            if (bitmap & 1) != 0:
                output_sign *= metric.signature()[i]
            i += 1
            bitmap >>= 1
        
    return output_sign

def geometric_product_blades(a_idx, a_weight, b_idx, b_weight, metric):
    a_idx = int(a_idx)
    b_idx = int(b_idx)
    bitmap = a_idx ^ b_idx

    sign = canonical_reordering_sign(a_idx, b_idx, metric)

    return bitmap, sign * a_weight * b_weight

def outer_product_blades(a_idx, a_weight, b_idx, b_weight, metric = None):
    a_idx = int(a_idx)
    b_idx = int(b_idx)
    if ((a_idx & b_idx) != 0):
        return 0, 0 
    else:
        return geometric_product_blades(a_idx, a_weight, b_idx, b_weight, metric)

def geometric_product(a, b, metric):
    A = metric.to_basis(a)
    nonzeroA = np.nonzero(A)[0]
    B = metric.to_basis(b)
    nonzeroB = np.nonzero(B)[0]

    result = np.zeros_like(A)
    for i in nonzeroA:
        for j in nonzeroB:
            blade_idx, blade_val = geometric_product_blades(i, A[i], j, B[j], metric)
            result[blade_idx] += blade_val

    return metric.to_metric(result)

# NOTE: use einsum to multiply with the op generated
def generate_geometric_product(metric):
    basis_size = metric.basis_dim()

    op = np.zeros(shape=(basis_size, basis_size, basis_size))
    for i in range(basis_size):
        it = np.zeros(shape=basis_size)
        it[i] = 1.0
        for j in range(basis_size):
            jt = np.zeros(shape=basis_size)
            jt[j] = 1.0

            op[i, j, :] = geometric_product(it, jt, metric)

    return op

class Metric:
    def __init__(self, metric_array):
        assert(is_symmetric(metric_array))

        self.metric_array = metric_array

        self.eig_vals, self.eig_vecs = np.linalg.eig(self.metric_array)
        self.inv_eig_vecs = np.linalg.inv(self.eig_vecs)

        self.geometric_product_tensor = generate_geometric_product(self)

    def get_geometric_product(self):
        return self.geometric_product_tensor

    def signature(self):
        return self.eig_vals

    def dims(self):
        return self.eig_vals.shape[0]

    def basis_dim(self):
        return 2 ** self.dims()

    def transform(self, x_idx, x_weight, basis):
        result = np.zeros(shape=self.basis_dim())
        result[0] = x_weight

        i = 0
        while x_idx != 0:
            if (x_idx & 1) == 1:
                tmp = np.zeros(shape=self.basis_dim())
                nonzeroCols = np.nonzero(basis[:, i])[0]
                for j in nonzeroCols:
                    basis_val = basis[j, i]
                    nonzeroResult = np.nonzero(result)[0]
                    for k in nonzeroResult:
                        idx, weight = outer_product_blades(k, result[k], 1 << j, basis_val)
                        tmp[idx] += weight
                result = tmp

            x_idx >>= 1
            i += 1
        
        return result

    def to_metric(self, x):
        result = np.zeros_like(x)
        nonzeroX = np.nonzero(x)[0]

        for i in nonzeroX:
            result = result + self.transform(i, x[i], self.eig_vecs)

        return result

    def to_basis(self, x):
        result = np.zeros_like(x)
        nonzeroX = np.nonzero(x)[0]

        for i in nonzeroX:
            result = result + self.transform(i, x[i], self.inv_eig_vecs)

        return result
