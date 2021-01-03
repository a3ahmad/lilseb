import numpy as np
from utils import bit_count, is_symmetric

def canonicalReorderingSignEuclidean(a_idx, b_idx):
    a = a_idx >> 1

    accum = 0
    while a != 0:
        accum = accum + bit_count(a & b_idx)
        a = a >> 1

    return 1.0 if (accum & 1) == 0 else -1.0

def canonicalReorderingSign(a_idx, b_idx, metric):
    bitmap = a_idx & b_idx
    output_sign = canonicalReorderingSignEuclidean(a_idx, b_idx)

    if metric != None:
        i = 0
        while bitmap != 0:
            if (bitmap & 1) != 0:
                output_sign *= metric.signature()[i]
            i += 1
            bitmap >>= 1
        
    return output_sign

def geometricProductBlades(a_idx, a_weight, b_idx, b_weight, metric):
    a_idx = int(a_idx)
    b_idx = int(b_idx)
    bitmap = a_idx ^ b_idx

    sign = canonicalReorderingSign(a_idx, b_idx, metric)

    return bitmap, sign * a_weight * b_weight

def outerProductBlades(a_idx, a_weight, b_idx, b_weight, metric = None):
    a_idx = int(a_idx)
    b_idx = int(b_idx)
    if ((a_idx & b_idx) != 0):
        return 0, 0 
    else:
        return geometricProductBlades(a_idx, a_weight, b_idx, b_weight, metric)

class Metric:
    def __init__(self, metric_array):
        assert(is_symmetric(metric_array))

        self.metric_array = metric_array

        self.eigVals, self.eigVecs = np.linalg.eig(self.metric_array)
        self.invEigVecs = np.linalg.inv(self.eigVecs)

    def signature(self):
        return self.eigVals

    def transform(self, x_idx, x_weight, basis):
        result = np.zeros(shape=2 ** self.eigVals.shape[0])
        result[0] = x_weight

        i = 0
        while x_idx != 0:
            if (x_idx & 1) == 1:
                tmp = np.zeros(shape=2 ** self.eigVals.shape[0])
                nonzeroCols = np.nonzero(basis[:, i])[0]
                for j in nonzeroCols:
                    basis_val = basis[j, i]
                    nonzeroResult = np.nonzero(result)[0]
                    for k in nonzeroResult:
                        idx, weight = outerProductBlades(k, result[k], 1 << j, basis_val)
                        tmp[idx] += weight
                result = tmp

            x_idx >>= 1
            i += 1
        
        return result

    def toMetric(self, x):
        result = np.zeros_like(x)
        nonzeroX = np.nonzero(x)[0]

        for i in nonzeroX:
            result = result + self.transform(i, x[i], self.eigVecs)

        return result

    def toBasis(self, x):
        result = np.zeros_like(x)
        nonzeroX = np.nonzero(x)[0]

        for i in nonzeroX:
            result = result + self.transform(i, x[i], self.invEigVecs)

        return result

def geometricProduct(a, b, metric):
    A = metric.toBasis(a)
    nonzeroA = np.nonzero(A)[0]
    B = metric.toBasis(b)
    nonzeroB = np.nonzero(B)[0]

    result = np.zeros_like(A)
    for i in nonzeroA:
        for j in nonzeroB:
            blade_idx, blade_val = geometricProductBlades(i, A[i], j, B[j], metric)
            result[blade_idx] += blade_val

    return metric.toMetric(result)

def generateGeometricProduct(a, metric):
    A = metric.toBasis(a)
    nonzeroA, _ = np.nonzero(A)
    # ANIS TODO: Use the following? Or force ones below?
    B = metric.toBasis(np.ones_like(a))
    nonzeroB, _ = np.nonzero(B)

    result = np.zeros_like(shape=(A.shape[0], A.shape[0]))
    for i in nonzeroA:
        for j in A.shape[0]:
            # ANIS TODO: Use the following? Or and entry of B above?
            blade_idx, blade_val = geometricProductBlades(i, A[i], j, 1.0, metric)
            # ANIS TODO: Transpose the following?
            result[i, blade_idx] += blade_val

    return metric.toMetric(result)
