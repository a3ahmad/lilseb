import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import init

from .algebra import Metric

import math

from icecream import ic


class BaseGALayer(nn.Module):
    def __init__(
            self,
            metric: Metric):
        super(BaseGALayer, self).__init__()

        self.metric = metric

        if not hasattr(self.metric, "torchGP"):
            setattr(self.metric, "torchGP", torch.tensor(
                self.metric.get_geometric_product(), dtype=torch.float))
        if not hasattr(self.metric, "torchReversionIdx"):
            setattr(self.metric, "torchReversionIdx",
                    (torch.tensor(self.metric.reversion) < 0).nonzero())
        if not hasattr(self.metric, "torchInvolutionIdx"):
            setattr(self.metric, "torchInvolutionIdx",
                    (torch.tensor(self.metric.involution) < 0).nonzero())
        if not hasattr(self.metric, "torchConjugationIdx"):
            setattr(self.metric, "torchConjugationIdx",
                    (torch.tensor(self.metric.conjugation) < 0).nonzero())


class SimpleEmbedToGA(BaseGALayer):
    def __init__(
            self,
            metric: Metric):
        super(SimpleEmbedToGA, self).__init__(metric)

    def forward(self, x):
        assert self.metric.dims() > x.shape[1]

        result = torch.zeros(
            size=(x.shape[0], 1, self.metric.basis_dim(), *tuple(x.shape[2:])), device=x.device)
        result[:, 0, 1:(x.shape[1] + 1), ...] = x
        return result


class SimpleGANormToFeatures(BaseGALayer):
    def __init__(
            self,
            metric: Metric):
        super(SimpleGANormToFeatures, self).__init__(metric)

        self.geomProd = nn.Parameter(
            self.metric.torchGP,
            requires_grad=False)

    def forward(self, x):
        # Compute the reversion of x for the norm
        revx = x.clone()
        revx[:, :, self.metric.torchReversionIdx, ...] = - \
            revx[:, :, self.metric.torchReversionIdx, ...]
        if len(x.shape) == 3:
            x = torch.einsum(
                'ijk,abi,abj->abk',
                self.geomProd, x, revx)
        elif len(x.shape) == 4:
            x = torch.einsum(
                'ijk,abic,abjc->abkc',
                self.geomProd, x, revx)
        elif len(x.shape) == 5:
            x = torch.einsum(
                'ijk,abicd,abjcd->abkcd',
                self.geomProd, x, revx)
        elif len(x.shape) == 6:
            x = torch.einsum(
                'ijk,abicde,abjcde->abkcde',
                self.geomProd, x, revx)
        return torch.sqrt(torch.sum(x, dim=2))


class ConvertLinearToGA(BaseGALayer):
    # Converts inputs of batch_size x num_input_features
    # to batch_size x num_output_features x ga_basis_dim
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(ConvertLinearToGA, self).__init__(metric)

        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features * metric.basis_dim())

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0], self.out_features, self.metric.basis_dim())


class ConvertGAToLinear(BaseGALayer):
    # Converts inputs of batch_size x num_input_features x ga_basis_dim
    # to batch_size x num_output_features
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int):
        super(ConvertGAToLinear, self).__init__(metric)

        self.in_features = in_features
        self.layer = nn.Linear(in_features * metric.basis_dim(), out_features)

    def forward(self, x):
        return self.layer(
            x.view(
                x.shape[0],
                self.in_features * self.metric.basis_dim()))


class Convert1dToGA(BaseGALayer):
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
        super(Convert1dToGA, self).__init__(metric)

        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(),
                               kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2])


class ConvertGATo1d(BaseGALayer):
    # Converts inputs from NCGW to NCW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int):
        super(Convert1dToGA, self).__init__(metric)

        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3]))


class Convert2dToGA(BaseGALayer):
    # Converts inputs from NCHW to NCGHW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert2dToGA, self).__init__(metric)

        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(),
                               kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2],
            result.shape[3])


class ConvertGATo2d(BaseGALayer):
    # Converts inputs from NCGHW to NCHW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int):
        super(Convert2dToGA, self).__init__(metric)

        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3],
            x.shape[4]))


class Convert3dToGA(BaseGALayer):
    # Converts inputs from NCDHW to NCGDHW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_3_t,
            stride: _size_3_t = 1,
            padding: _size_3_t = 0,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Convert3dToGA, self).__init__(metric)

        self.out_channels = out_channels
        self.layer = nn.Conv2d(in_channels, out_channels * metric.basis_dim(),
                               kernel_size, stride, padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        result = self.layer(x)
        return result.view(
            result.shape[0],
            self.out_features,
            self.metric.basis_dim(),
            result.shape[2],
            result.shape[3],
            result.shape[4])


class ConvertGATo3d(BaseGALayer):
    # Converts inputs from NCGDHW to NCDHW
    def __init__(
            self,
            metric: Metric,
            in_channels: int,
            out_channels: int):
        super(ConvertGATo3d, self).__init__(metric)

        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels * metric.basis_dim(), out_channels)

    def forward(self, x):
        return self.layer(x.view(
            x.shape[0],
            self.in_channels * self.metric.basis_dim(),
            x.shape[3],
            x.shape[4],
            x.shape[5]))


class GAFlatten(BaseGALayer):
    def __init__(
            self,
            metric: Metric):
        super(GAFlatten, self).__init__(metric)

    def forward(self, x):
        x = x.transpose(1, 2).flatten(2, -1).transpose(1, 2)
        return x


class GPLinear(BaseGALayer):
    def __init__(
            self,
            metric: Metric,
            in_features: int,
            out_features: int,
            bias: bool = True,
            versor: bool = False):
        super(GPLinear, self).__init__(metric)

        self.geomProd = nn.Parameter(
            self.metric.torchGP,
            requires_grad=False)
        self.W = nn.Parameter(
            torch.empty(
                size=(in_features, out_features, metric.basis_dim())),
            requires_grad=True)
        self.b = nn.Parameter(
            torch.empty(
                size=(out_features, metric.basis_dim())),
            requires_grad=True) if bias else None
        self.bias = bias
        self.versor = versor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.named_buffers, -bound, bound)

    def forward(self, x):
        result = torch.einsum(
            # i = input GA 'a' component
            # j = input GA 'b' component
            # k = output GA component
            # b = batch number
            # p = input feature
            # o = output feature
            'ijk,poi,bpj->bok',
            self.geomProd, self.W, x)
        if self.versor:
            revW = self.W.clone()
            revW[..., self.metric.torchReversionIdx] = - \
                revW[..., self.metric.torchReversionIdx]
            WSqNorm = torch.einsum(
                'ijk,poi,poj->pok',
                self.geomProd, self.W, revW)
            revW = revW / WSqNorm[:, :, 0, ...]

            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # p = input feature
                # o = output feature
                'ijk,bpi,poj->bok',
                self.geomProd, result, revW)
        if self.bias:
            result = result + self.b
        return result


# PyTorch's padding functions don't work with our representations, only NCW, NCHW, NCDHW
def ga_pad(t, padding, padding_mode, value = None):
    p = padding
    assert len(t.shape) == 3 + len(p)

    # Compute the size of the padded tensor
    padded_shape = t.shape[:3]
    for _r, _p in zip(t.shape[3:], p):
        padded_shape = padded_shape + (_r + 2 * _p,)

    # Create the padded tensor
    if padding_mode == 'constant':
        if value is not None:
            padded = torch.full(size=padded_shape, fill_value=value, device=t.device)
        else:
            padded = torch.zeros(size=padded_shape, device=t.device)
    else:
        padded = torch.empty(size=padded_shape, device=t.device)

    # Copy in source tensor
    orig_slice = (...,)
    for _p in p:
        orig_slice = orig_slice + (slice(_p, -_p),)
    padded[orig_slice] = t

    # pad
    if padding_mode == 'reflect':
        # ANIS TODO
        pass
    elif padding_mode == 'replicate':
        bcg = (slice(None), slice(None), slice(None),)
        for d in range(len(p)):
            padded[bcg + (slice(None, p[d]),) + orig_slice[(3+d+1):]] = padded[bcg + (p[d],) + orig_slice[(3+d+1):]].unsqueeze(3+d)
            padded[bcg + (slice(p[d], None),) + orig_slice[(3+d+1):]] = padded[bcg + (-p[d]-1,) + orig_slice[(3+d+1):]].unsqueeze(3+d)
            bcg = bcg + (slice(None),)
    elif padding_mode == 'circular':
        # ANIS TODO
        pass

    return padded


class GPConv1d(BaseGALayer):
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
            versor: bool = False):
        super(GPConv1d, self).__init__(metric)

        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.padding_mode = padding_mode
        self.geomProd = nn.Parameter(
            self.metric.torchGP,
            requires_grad=False)
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    self.kernel_size[0],
                    in_channels,
                    out_channels,
                    metric.basis_dim())),
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(
                torch.empty(size=(1, out_channels, metric.basis_dim(), 1)),
                requires_grad=True)
        self.versor = versor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x = ga_pad(x, self.padding, self.padding_mode)
        x = x.unfold(3, self.W.shape[0], self.stride[0])
        result = torch.einsum(
            # i = input GA 'a' component
            # j = input GA 'b' component
            # k = output GA component
            # b = batch number
            # c = image channel
            # w = image width
            # v = convolution width
            # o = output channel
            'ijk,vcoi,bcjwv->bokh',
            self.geomProd, self.W, x)
        if self.versor:
            revW = self.W.clone()
            revW[..., self.metric.torchReversionIdx] = - \
                revW[..., self.metric.torchReversionIdx]
            WSqNorm = torch.einsum(
                'ijk,vcoi,vcoj->vcok',
                self.geomProd, self.W, revW)
            revW = revW / WSqNorm[:, :, 0, ...]

            result = ga_pad(result, self.padding, self.padding_mode)
            result = result.unfold(3, revW.shape[0], self.stride[0])
            result = torch.einsum(
                # i = input GA 'a' component
                # j = input GA 'b' component
                # k = output GA component
                # b = batch number
                # c = image channel
                # w = image width
                # v = convolution width
                # o = output channel
                'ijk,bcjwv,vcoj->bokh',
                self.geomProd, result, revW)
        if self.b is not None:
            result = result + self.b
        return result


class GPConv2d(BaseGALayer):
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
            versor: bool = False):
        super(GPConv2d, self).__init__(metric)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.padding_mode = padding_mode
        self.geomProd = nn.Parameter(
            self.metric.torchGP,
            requires_grad=False)
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    self.kernel_size[0],
                    self.kernel_size[1],
                    in_channels,
                    out_channels,
                    metric.basis_dim())),
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(
                torch.empty(size=(1, out_channels, metric.basis_dim(), 1, 1)),
                requires_grad=True)
        self.versor = versor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x = ga_pad(x, self.padding, self.padding_mode)
        x = x.unfold(3, self.W.shape[0], self.stride[0])
        x = x.unfold(4, self.W.shape[1], self.stride[1])
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
            'ijk,lvcoi,bcjhwlv->bokhw',
            self.geomProd, self.W, x)
        if self.versor:
            revW = self.W.clone()
            revW[..., self.metric.torchReversionIdx] = - \
                revW[..., self.metric.torchReversionIdx]
            WSqNorm = torch.einsum(
                'ijk,lvcoi,lvcoj->lvcok',
                self.geomProd, self.W, revW)
            revW = revW / WSqNorm[:, :, 0, ...]

            result = ga_pad(result, self.padding, self.padding_mode)
            result = result.unfold(3, revW.shape[0], self.stride[0])
            result = result.unfold(4, revW.shape[1], self.stride[1])
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
                self.geomProd, result, revW)
        if self.b is not None:
            result = result + self.b
        return result


class GPConv3d(BaseGALayer):
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
            versor: bool = False):
        super(GPConv3d, self).__init__(metric)

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.padding_mode = padding_mode
        self.geomProd = nn.Parameter(
            self.metric.torchGP,
            requires_grad=False)
        self.W = nn.Parameter(
            torch.empty(
                size=(
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.kernel_size[2],
                    in_channels,
                    out_channels,
                    metric.basis_dim())),
            requires_grad=True)
        if bias:
            self.b = nn.Parameter(
                torch.empty(size=(1, out_channels, metric.basis_dim(), 1, 1, 1)),
                requires_grad=True)
        self.versor = versor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x = ga_pad(x, self.padding, self.padding_mode)
        x = x.unfold(3, self.W.shape[0], self.stride[0])
        x = x.unfold(4, self.W.shape[1], self.stride[1])
        x = x.unfold(5, self.W.shape[2], self.stride[2])
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
            'ijk,mlvcoi,bcjdhwmlv->bokdhw',
            self.geomProd, self.W, x)
        if self.versor:
            revW = self.W.clone()
            revW[..., self.metric.torchReversionIdx] = - \
                revW[..., self.metric.torchReversionIdx]
            WSqNorm = torch.einsum(
                'ijk,mlvcoi,mlvcoj->mlvcok',
                self.geomProd, self.W, revW)
            revW = revW / WSqNorm[:, :, 0, ...]

            result = ga_pad(result, self.padding, self.padding_mode)
            result = result.unfold(3, revW.shape[0], self.stride[0])
            result = result.unfold(4, revW.shape[1], self.stride[1])
            result = result.unfold(5, revW.shape[2], self.stride[2])
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
                self.geomProd, result, revW)
        if self.b is not None:
            result = result + self.b
        return result
