import torch
import torch.nn as nn
from typing import Any, List


def _make_scratch(in_shape: List[int], out_shape: int, groups: int=1, expand: bool=False) -> nn.Module:
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    
    return scratch


class ResidualConvUnit(nn.Module):
    """
    Residual convolution module.
    """
    def __init__(self, features: int, activation: nn.Module, bn: bool):
        """
        Init.

        Args:
            features (int): The number of features.
        """
        super(ResidualConvUnit, self).__init__()

        self.bn = bn
        
        self.groups = 1
        
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        
        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (tensor): The input.

        Returns:
            tensor: The output.
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)
        
        if self.groups > 1:
            out = self.conv_merge(out)
        
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block.
    """
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool=False,
        bn: bool=False,
        expand: bool=False,
        align_corners: bool=True,
        size: Any=None,
    ):
        """
        Init.
        
        Args:
            features (int): The number of features.
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size
    
    def forward(self, *xs: Any, size: Any=None) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            tensor: The output.
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        
        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
