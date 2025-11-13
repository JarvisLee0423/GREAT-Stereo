# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        act_layer: Callable[..., nn.Module]=None,
        drop: float=0.0,
        bias: bool=True,
    ):
        super(SwiGLUFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2

        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU
    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        act_layer: Callable[..., nn.Module]=None,
        drop: float=0.0,
        bias: bool=True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super(SwiGLUFFNFused, self).__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
