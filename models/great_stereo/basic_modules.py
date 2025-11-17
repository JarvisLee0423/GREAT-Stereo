import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple


def disparity_regression(inputs: torch.Tensor, max_disp: int) -> torch.Tensor:
    assert len(inputs.shape) == 4
    disp_values = torch.arange(0, max_disp, dtype=inputs.dtype, device=inputs.device)
    disp_values = disp_values.view(1, max_disp, 1, 1)

    return torch.sum(inputs * disp_values, 1, keepdim=True)


def context_upsample(disp_low: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode="nearest").reshape(b, 9, h * 4, w * 4)
    disp = (disp_unfold * up_weights).sum(1)

    return disp


def compute_scale_shift(mono_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor=None) -> Tuple[int]:
    flattened_depth_maps = mono_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (mono_depth > 1e-2) & (mono_depth > threshold_10_percent)
    
    mono_depth_flat = mono_depth[mask]
    gt_depth_flat = gt_depth[mask]

    X = torch.stack([mono_depth_flat, torch.ones_like(mono_depth_flat)], dim=1)
    y = gt_depth_flat

    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)

    scale, shift = params[0].item(), params[1].item()

    return scale, shift


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, bn: bool=True, relu: bool=True, **kwargs: Any):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.conv(inputs)
        if self.use_bn:
            outs = self.bn(outs)
        if self.relu:
            outs = nn.LeakyReLU()(outs)
        
        return outs


class BasicConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, bn: bool=True, relu: bool=True, **kwargs):
        super(BasicConvReLU, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv(inputs)
        if self.use_bn:
            inputs = self.bn(inputs)
        if self.relu:
            inputs = F.relu(inputs, inplace=True)
        
        return inputs


class BasicConvIN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, IN: bool=True, relu: bool=True, **kwargs: Any):
        super(BasicConvIN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.conv(inputs)
        if self.use_in:
            outs = self.IN(outs)
        if self.relu:
            outs = nn.LeakyReLU()(outs)
        
        return outs


class Conv2x(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, concat: bool=True, keep_concat: bool=True, bn: bool=True, relu: bool=True, keep_dispc: bool=False):
        super(Conv2x, self).__init__()
        
        self.concat = concat
        self.id_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        
        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        
        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
    
    def forward(self, inputs: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        outs = self.conv1(inputs)
        if outs.shape != rem.shape:
            outs = F.interpolate(
                outs,
                size=(rem.shape[-2], rem.shape[-1]),
                mode="nearest",
            )
        if self.concat:
            outs = torch.cat((outs, rem), 1)
        else:
            outs = outs + rem
        outs = self.conv2(outs)

        return outs


class Conv2xReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, concat: bool=True, bn: bool=True, relu: bool=True, mdconv: bool=False):
        super(Conv2xReLU, self).__init__()

        self.concat = concat
        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConvReLU(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConvReLU(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConvReLU(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inputs: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        assert (inputs.size() == rem.size()), "The shape of inputs and rem must be the same."

        if self.concat:
            inputs = torch.cat((inputs, rem), 1)
        else:
            inputs = inputs + rem
        inputs = self.conv2(inputs)

        return inputs


class Conv2xIN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool=False, is_3d: bool=False, concat: bool=True, keep_concat: bool=True, IN: bool=True, relu: bool=True, keep_dispc: bool=False):
        super(Conv2xIN, self).__init__()

        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        
        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConvIN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConvIN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        
        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConvIN(out_channels * 2, out_channels * mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConvIN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
    
    def forward(self, inputs: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        outs = self.conv1(inputs)
        if outs.shape != rem.shape:
            outs = F.interpolate(
                outs,
                size=(rem.shape[-2], rem.shape[-1]),
                mode="nearest",
            )
        if self.concat:
            outs = torch.cat((outs, rem), 1)
        else:
            outs = outs + rem
        outs = self.conv2(outs)

        return outs
