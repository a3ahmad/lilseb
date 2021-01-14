import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .algebra import Metric


class SimpleGAEmbedding(nn.Module):
    def __init__(
            self,
            metric: Metric):
        super(SimpleGAEmbedding, self).__init__()

        self.metric = metric

    def forward(self, x):
        assert self.metric.dims() > x.shape[1]

        result = torch.zeros(size=(x.shape[0], 1, self.metric.basis_dim(), *tuple(x.shape[2:])))
        result[:, 0, 1:(x.shape[1] + 1), ...] = x
        return result


class ConvertLinearToGA(nn.Module):
    # Converts inputs of batch_size x num_input_features
    # to batch_size x num_output_features x ga_basis_dim
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(ConvertLinearToGA, self).__init__()

        self.metric = metric
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features * metric.basis_dim())

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0], self.out_features, self.metric.basis_dim())


class ConvertGAToLinear(nn.Module):
    # Converts inputs of batch_size x num_input_features x ga_basis_dim
    # to batch_size x num_output_features
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(ConvertGAToLinear, self).__init__()

        self.metric = metric
        self.in_features = in_features
        self.layer = nn.Linear(in_features * metric.basis_dim(), out_features)

    def forward(self, x):
        return self.layer(
                x.view(
                    x.shape[0],
                    self.in_features * self.metric.basis_dim()))


class Convert1DToGA(nn.Module):
    # Converts inputs from NCW to NCGW, were G is the geometric algebra
    # basis dimension
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: _size_1_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert1DToGA, self).__init__()

        self.metric = metric
        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(), kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2])


class ConvertGATo1D(nn.Module):
    # Converts inputs from NCGW to NCW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: _size_1_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert1DToGA, self).__init__()

        self.metric = metric
        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels, kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3]))


class Convert2DToGA(nn.Module):
    # Converts inputs from NCHW to NCGHW
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert2DToGA, self).__init__()

        self.metric = metric
        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(), kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2],
            result.shape[3])


class ConvertGATo2D(nn.Module):
    # Converts inputs from NCGHW to NCHW
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(Convert2DToGA, self).__init__()

        self.metric = metric
        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels, kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3],
            x.shape[4]))


class Convert3DToGA(nn.Module):
    # Converts inputs from NCDHW to NCGDHW
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_3_t,
            stride: _size_3_t = 1,
            padding: _size_3_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert3DToGA, self).__init__()

        self.metric = metric
        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(), kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2],
            result.shape[3],
            result.shape[4])

class ConvertGATo3D(nn.Module):
    # Converts inputs from NCGDHW to NCDHW
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(ConvertGATo3D, self).__init__()

        self.metric = metric
        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels, kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3],
            x.shape[4],
            x.shape[5]))


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
        self.W = nn.Parameter(
            torch.empty(
                size=(in_features, out_features, metric.basis_dim())),
            requires_grad=True)
        self.b = nn.Parameter(
            torch.empty(
                size=(out_features, metric.basis_dim())),
            requires_grad=True) if bias else None
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # p = input feature
                # o = output feature
                'ijk,bpi,poj->bok',
                self.metric.get_geometric_product(), x, self.W)
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
            padding: _size_1_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros',
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
            self.b = nn.Parameter(
                torch.empty(size=(out_channels, metric.basis_dim())),
                requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            x = F.pad(x, self.padding, self.padding_mode)
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # c = image channel
                # w = image width
                # v = convolution width
                # o = output channel
                'ijk,bciwv,vcoj->bokh',
                self.metric.get_geometric_product(), x, self.W)
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
            padding: _size_2_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros',
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
            self.b = nn.Parameter(
                torch.empty(size=(out_channels, metric.basis_dim())),
                requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            x = F.pad(x, self.padding, self.padding_mode)
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            x = x.unfold(3, self.W.shape[1], self.stride[1])
            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # c = image channel
                # h = image height
                # w = image width
                # l = convolution height
                # v = convolution width
                # o = output channel
                'ijk,bcihwlv,lvcoj->bokhw',
                self.metric.get_geometric_product(), x, self.W)
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
            padding: _size_3_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros',
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
            self.b = nn.Parameter(
                torch.empty(size=(out_channels, metric.basis_dim())),
                requires_grad=True)
        self.versor = versor

    def forward(self, x):
        result = None
        if not self.versor:
            x = F.pad(x, self.padding, self.padding_mode)
            x = x.unfold(2, self.W.shape[0], self.stride[0])
            x = x.unfold(3, self.W.shape[1], self.stride[1])
            x = x.unfold(4, self.W.shape[2], self.stride[2])
            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # c = image channel
                # d = image depth
                # h = image height
                # w = image width
                # m = convolution depth
                # l = convolution height
                # v = convolution width
                # o = output channel
                'ijk,bcidhwmlv,mlvcoj->bokdhw',
                self.metric.get_geometric_product(), x, self.W)
            if self.b is not None:
                result = result + self.b
        return result
