import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .algebra import Metric

class GPLinear(nn.Module):
    def __init__(
            self, 
            metric: Metric,
            in_features: int, 
            out_features: int, 
            bias: bool = True, 
            versor: bool = False):  # Unsupported
        super(GPLinear, self).__init__()

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
            metric: Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_1_t, 
            stride: _size_1_t = 1, 
            padding: _size_1_t = 0,         # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        super(GPConv1D, self).__init__()
        
        self.metric = metric
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    kernel_size[0], 
                    in_channels, 
                    out_channels, 
                    metric.basis_dim())), 
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.empty(size=(out_channels, metric.basis_dim())), requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            # TODO: Pad
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            result = torch.einsum('ijk,bcwki,kcoj->bohk', self.metric.geometricProductTensor, x, self.W)
            if self.b is not None:
                result = result + self.b
        return result

class GPConv2D(nn.Module):
    def __init__(
            self, 
            metric: Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_2_t, 
            stride: _size_2_t = 1, 
            padding: _size_2_t = 0,         # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        super(GPConv2D, self).__init__()
        
        self.metric = metric
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    kernel_size[0], 
                    kernel_size[1], 
                    in_channels, 
                    out_channels, 
                    metric.basis_dim())), 
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.empty(size=(out_channels, metric.basis_dim())), requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            # TODO: Pad
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            x = x.unfold(3, self.W.shape[1], self.stride[1])
            result = torch.einsum('ijk,bchwlki,lkcoj->bohwk', self.metric.geometricProductTensor, x, self.W)
            if self.b is not None:
                result = result + self.b
        return result

class GPConv3D(nn.Module):
    def __init__(
            self, 
            metric: Metric,
            in_channels: int, 
            out_channels: int, 
            kernel_size: _size_3_t, 
            stride: _size_3_t = 1, 
            padding: _size_3_t = 0,         # Unsupported
            bias: bool = True, 
            padding_mode: str = 'zeros',    # Unsupported
            versor: bool = False):          # Unsupported
        super(GPConv3D, self).__init__()
        
        self.metric = metric
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    kernel_size[0], 
                    kernel_size[1], 
                    kernel_size[2], 
                    in_channels, 
                    out_channels, 
                    metric.basis_dim())), 
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.empty(size=(out_channels, metric.basis_dim())), requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            # TODO: Pad
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            x = x.unfold(3, self.W.shape[1], self.stride[1])
            x = x.unfold(4, self.W.shape[2], self.stride[2])
            result = torch.einsum('ijk,bcdhwmlki,mlkcoj->bodhwk', self.metric.geometricProductTensor, x, self.W)
            if self.b is not None:
                result = result + self.b
        return result
