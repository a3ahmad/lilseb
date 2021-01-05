import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

import algebra

class GPLinear(nn.Module):
    def __init__(
            self, 
            metric: algebra.Metric,
            in_features: int, 
            out_features: int, 
            bias: bool = True, 
            versor: bool = False):  # Unsupported
        self.metric = metric
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features, metric.basis_dim())), requires_grad=True)
        self.b = nn.Parameter(torch.empty(size=(out_features, metric.basis_dim())), requires_grad=True) if bias else None
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            result = torch.einsum('ijk,bpi,poj->bok', self.metric.geometricProductTensor, x, self.W)
            if self.bias:
                result = result + self.b
        return result

class GPConv1D(nn.Module):
    def __init__(
            self, 
            metric: algebra.Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_1_t, 
            stride: _size_1_t = 1, 
            padding: _size_1_t = 0,         # Unsupported
            dilation: _size_1_t = 1,        # Unsupported
            groups: int = 1,                # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        # ANIS TODO:
        pass

class GPConv2D(nn.Module):
    def __init__(
            self, 
            metric: algebra.Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_2_t, 
            stride: _size_2_t = 1, 
            padding: _size_2_t = 0,         # Unsupported
            dilation: _size_2_t = 1,        # Unsupported
            groups: int = 1,                # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        # ANIS TODO:
        pass

class GPConv3D(nn.Module):
    def __init__(
            self, 
            metric: algebra.Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_3_t, 
            stride: _size_3_t = 1, 
            padding: _size_3_t = 0,         # Unsupported
            dilation: _size_3_t = 1,        # Unsupported
            groups: int = 1,                # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        # ANIS TODO:
        pass
