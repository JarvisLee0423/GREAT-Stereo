import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Union
from einops import rearrange
from models.great_stereo.attentions import *


def to_3d(inputs: torch.Tensor, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "b c h w -> (b h) w c")
    else:
        return rearrange(inputs, "b c h w -> b (h w) c")


def to_4d(inputs: torch.Tensor, b: Union[int, float], h: Union[int, float], w: Union[int, float]=None, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "(b h) w c -> b c h w", b=b, h=h)
    else:
        return rearrange(inputs, "b (h w) c -> b c h w", h=h, w=w)


class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int=2,
        receptive_aug: bool=False,
        norm_layer: Any=nn.LayerNorm,
        act_func: Any=nn.SiLU,
    ):
        super(FeedForwardLayer, self).__init__()

        self.receptive_aug = receptive_aug

        self.norm = norm_layer(in_channels)
        self.in_conv = nn.Conv2d(in_channels, expansion * in_channels, kernel_size=1)
        self.act = act_func()

        if self.receptive_aug is not None:
            if self.receptive_aug == "pool":
                self.max_pool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
                self.max_pool_5 = nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=2)
                self.max_pool_7 = nn.MaxPool2d(kernel_size=(7, 7), stride=1, padding=3)

                self.avg_pool_3 = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)
                self.avg_pool_5 = nn.AvgPool2d(kernel_size=(5, 5), stride=1, padding=2)
                self.avg_pool_7 = nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
            elif self.receptive_aug == "conv":
                self.dw_conv_3 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels, kernel_size=3, stride=1, padding=1, groups=expansion * in_channels)
                self.dw_conv_5 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels, kernel_size=5, stride=1, padding=2, groups=expansion * in_channels)
                self.dw_conv_7 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels, kernel_size=7, stride=1, padding=3, groups=expansion * in_channels)
            else:
                raise ValueError(f"{self.receptive_aug} is not supported! Valid setting is 'pool' or 'conv'.")
            self.pointwise_conv = nn.Sequential(
                nn.Conv2d(in_channels=3 * expansion * in_channels, out_channels=expansion * in_channels, kernel_size=1),
                act_func(),
            )
        else:
            print("No receptive augmentation.")
        
        self.out_conv = nn.Conv2d(expansion * in_channels, out_channels, kernel_size=1)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.shape
        inputs = to_4d(self.norm(to_3d(inputs)), b, h)
        inputs = self.in_conv(inputs)
        inputs = self.act(inputs)
        if self.receptive_aug is not None:
            if self.receptive_aug == "pool":
                max_3 = self.max_pool_3(inputs)
                max_5 = self.max_pool_5(inputs)
                max_7 = self.max_pool_7(inputs)
                max_feats = torch.cat([max_3, max_5, max_7], dim=1)

                avg_3 = self.avg_pool_3(inputs)
                avg_5 = self.avg_pool_5(inputs)
                avg_7 = self.avg_pool_7(inputs)
                avg_feats = torch.cat([avg_3, avg_5, avg_7], dim=1)

                inputs = inputs + self.pointwise_conv(max_feats + avg_feats)
            else:
                feats_3 = self.dw_conv_3(inputs)
                feats_5 = self.dw_conv_5(inputs)
                feats_7 = self.dw_conv_7(inputs)

                inputs = inputs + self.pointwise_conv(torch.cat([feats_3, feats_5, feats_7], dim=1))

        inputs = self.out_conv(inputs)

        return inputs


class AttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int=8,
        dropout: float=0.0,
        pre_norm: bool=True,
        norm_layer: Any=nn.LayerNorm,
        sink_competition: bool=False,
        qkv_bias: bool=True,
        eps: float=1e-6,
    ):
        super(AttentionLayer, self).__init__()

        self.eps = eps
        self.pre_norm = pre_norm
        self.sink_competition = sink_competition
        assert (hidden_channels % num_heads) == 0, "hidden_dim and num_heads are not divisible."
        self.scale = (hidden_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(in_channels if pre_norm else out_channels, eps=eps)
        self.norm_context = norm_layer(in_channels, eps=eps) if pre_norm else None

        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=qkv_bias)
        self.to_kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=qkv_bias)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> Tuple[torch.Tensor]:
        b, c, h, w = feats1.shape
        if self.pre_norm:
            feats1 = to_4d(self.norm(to_3d(feats1)), b, h)
            feats2 = to_4d(self.norm_context(to_3d(feats2)), b, h)
        
        query = to_3d(self.to_q(feats1))
        key, value = to_3d(self.to_kv(feats2)).chunk(2, dim=-1)

        query, key, value = map(
            lambda t: rearrange(t, "bh w (n c) -> (bh n) w c", n=self.num_heads),
            (query, key, value),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale

        if self.sink_competition:
            raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
            raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
        else:
            raw_attn = F.softmax(similarity_matrix, dim=-1)
        
        attn = self.dropout(raw_attn)

        out = torch.einsum("bij, bjd -> bid", attn, value)
        out = rearrange(out, "(bh n) w c -> bh w (n c)", n=self.num_heads)
        out = to_3d(self.to_out(to_4d(out, b, h)))
        if not self.pre_norm:
            out = self.norm(out)
        out = to_4d(out, b, h)

        return out, similarity_matrix


class VolumeAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int=8,
        dropout: float=0.0,
        pre_norm: bool=True,
        norm_layer: Any=nn.LayerNorm,
        sink_competition: bool=False,
        qkv_bias: bool=True,
        eps: float=1e-6,
    ):
        super(VolumeAttentionLayer, self).__init__()

        self.eps = eps
        self.pre_norm = pre_norm
        self.sink_competition = sink_competition
        assert (hidden_channels % num_heads) == 0, "hidden_dim and num_heads are not divisible."
        self.scale = (hidden_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(in_channels if pre_norm else out_channels, eps=eps)
        self.norm_context = norm_layer(in_channels, eps=eps) if pre_norm else None

        self.to_q = nn.Linear(in_channels, hidden_channels, bias=qkv_bias)
        self.to_kv = nn.Linear(in_channels, hidden_channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, volume: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor]:
        b, c, d, h, w = volume.shape
        if self.pre_norm:
            volume = self.norm(rearrange(volume, "b c d h w -> (b h w) d c"))
            feats = self.norm_context(rearrange(feats, "b c h w -> (b h w) c").unsqueeze(-2))
        
        query = self.to_q(volume)
        key, value = self.to_kv(feats).chunk(2, dim=-1)

        query, key, value = map(
            lambda t: rearrange(t, "bhw d (n c) -> (bhw n) d c", n=self.num_heads),
            (query, key, value),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale

        if self.sink_competition:
            raw_attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            raw_attn = raw_attn / torch.sum(raw_attn, dim=(-2,), keepdim=True)
        else:
            raw_attn = F.softmax(similarity_matrix, dim=-2)
        attn = self.dropout(raw_attn)

        out = torch.einsum("bij, bjd -> bid", attn, value)
        out = rearrange(out, "(bhw n) d c -> (bhw) d (n c)", n=self.num_heads)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)
        out = rearrange(out, "(b h w) d c -> b c d h w", b=b, h=h, w=w)
        attn = rearrange(attn, "(b h w) d c -> b c d h w", b=b, h=h, w=w)

        return out, attn


class OutlookAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int=3,
        padding: int=1,
        stride: int=1,
        num_heads: int=8,
        dropout: float=0.0,
        pre_norm: bool=True,
        norm_layer: Any=nn.LayerNorm,
        sink_competition: bool=False,
        qkv_bias: bool=True,
        eps: float=1e-6,
    ):
        super(OutlookAttentionLayer, self).__init__()

        self.eps = eps
        self.pre_norm = pre_norm
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.sink_competition = sink_competition
        assert (hidden_channels % num_heads) == 0, "hidden_dim and num_heads are not divisible."
        self.scale = (hidden_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(in_channels if pre_norm else out_channels, eps=eps)

        self.to_v = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=qkv_bias)
        self.to_attn = nn.Conv2d(in_channels, kernel_size ** 4 * num_heads, kernel_size=1, bias=qkv_bias)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor]:
        b, c, h, w = feats.shape
        if self.pre_norm:
            feats = to_4d(self.norm(to_3d(feats, dim=2)), b, h, w, dim=2)
        
        value = self.to_v(feats)
        hh, ww = math.ceil(h / self.stride), math.ceil(w / self.stride)
        value = self.unfold(value).reshape(b, self.num_heads, c // self.num_heads, self.kernel_size * self.kernel_size, hh * ww).permute(0, 1, 4, 3, 2) # [B, Head, N, K*K, C/H].
        similarity_matrix = self.to_attn(self.pool(feats)).reshape(b, hh * ww, self.num_heads, self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4) # [B, Head, N, K*K, K*K]
        similarity_matrix = similarity_matrix * self.scale

        if self.sink_competition:
            raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
            raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
        else:
            raw_attn = F.softmax(similarity_matrix, dim=-1)
        
        attn = self.dropout(raw_attn)

        out = torch.einsum("bhnij, bhnjd -> bhnid", attn, value).permute(0, 1, 4, 3, 2).reshape(b, c * self.kernel_size * self.kernel_size, hh * ww)
        out = F.fold(out, output_size=(h, w), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        out = to_3d(self.to_out(out), dim=2)
        if not self.pre_norm:
            out = self.norm(out)
        out = to_4d(out, b, h, w, dim=2)

        return out, similarity_matrix


class WindowAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int=8,
        dropout: float=0.0,
        pre_norm: bool=True,
        norm_layer: Any=nn.LayerNorm,
        sink_competition: bool=False,
        qkv_bias: bool=True,
        eps: float=1e-6,
    ):
        super(WindowAttentionLayer, self).__init__()

        self.eps = eps
        self.pre_norm = pre_norm
        self.sink_competition = sink_competition
        assert (hidden_channels % num_heads) == 0, "hidden_dim and num_heads are not divisible."
        self.scale = (hidden_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(in_channels if pre_norm else out_channels, eps=eps)
        self.norm_context = norm_layer(in_channels, eps=eps) if pre_norm else None

        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=qkv_bias)
        self.to_kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=qkv_bias)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor, num_splits: int, with_shift: bool, swin_1d: bool) -> torch.Tensor:
        b, c, h, w = feats1.shape

        if swin_1d:
            if self.pre_norm:
                feats1 = to_4d(self.norm(to_3d(feats1, dim=1)), b, h, w, dim=1)
                feats2 = to_4d(self.norm_context(to_3d(feats2, dim=1)), b, h, w, dim=1)
            
            query = to_3d(self.to_q(feats1), dim=1)
            key, value = to_3d(self.to_kv(feats2), dim=1).chunk(2, dim=-1)

            query, key, value = map(
                lambda t: rearrange(t, "bh w (n c) -> (bh n) w c", n=self.num_heads),
                (query, key, value),
            )

            # Compute the window attention based on Swin-Transformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py.
            window_width = w // num_splits

            shifted_window_attn_mask = generate_shift_window_attn_mask(
                height=None,
                width=w,
                window_height=None,
                window_width=window_width,
                shift_height=None,
                shift_width=window_width // 2,
                device=feats1.device,
                swin_1d=swin_1d,
            ) # [K, W/K, W/K].

            if with_shift:
                shift_width = window_width // 2

                query = torch.roll(query, shifts=-shift_width, dims=1)
                key = torch.roll(key, shifts=-shift_width, dims=1)
                value = torch.roll(value, shifts=-shift_width, dims=1)
            
            query = split_feature(query, num_splits=num_splits, swin_1d=swin_1d) # [B*H*N*K, W/K, C].
            key = split_feature(key, num_splits=num_splits, swin_1d=swin_1d)
            value = split_feature(value, num_splits=num_splits, swin_1d=swin_1d)

            similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale
            if with_shift:
                similarity_matrix += shifted_window_attn_mask.repeat(b * h * self.num_heads, 1, 1) # [B*H*N*K, W/K, W/K].

            if self.sink_competition:
                raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
                raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
            else:
                raw_attn = F.softmax(similarity_matrix, dim=-1)
            
            attn = self.dropout(raw_attn)

            out = torch.einsum("bij, bjd -> bid", attn, value)
            out = merge_splits(out, h, num_splits, swin_1d=swin_1d)
            if with_shift:
                out = torch.roll(out, shifts=shift_width, dims=2)
            out = rearrange(out, "(b n) h w c -> b (n c) h w", n=self.num_heads)
            out = to_3d(self.to_out(out), dim=1)
            if not self.pre_norm:
                out = self.norm(out)
            out = to_4d(out, b, h, w, dim=1)
        else:
            if self.pre_norm:
                feats1 = to_4d(self.norm(to_3d(feats1, dim=2)), b, h, w, dim=2)
                feats2 = to_4d(self.norm_context(to_3d(feats2, dim=2)), b, h, w, dim=2)
            
            query = to_3d(self.to_q(feats1), dim=2)
            key, value = to_3d(self.to_kv(feats2), dim=2).chunk(2, dim=-1)

            query, key, value = map(
                lambda t: rearrange(t, "b hw (n c) -> (b n) hw c", n=self.num_heads),
                (query, key, value),
            )
            
            # Compute the window attention based on Swin-Transformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py.
            window_height = h // num_splits
            window_width = w // num_splits

            shifted_window_attn_mask = generate_shift_window_attn_mask(
                height=h,
                width=w,
                window_height=window_height,
                window_width=window_width,
                shift_height=window_height // 2,
                shift_width=window_width // 2,
                device=feats1.device,
                swin_1d=swin_1d,
            )

            query = query.view(b * self.num_heads, h, w, c // self.num_heads)
            key = key.view(b * self.num_heads, h, w, c // self.num_heads)
            value = value.view(b * self.num_heads, h, w, c // self.num_heads)

            if with_shift:
                shift_height = window_height // 2
                shift_width = window_width // 2

                query = torch.roll(query, shifts=(-shift_height, -shift_width), dims=(1, 2))
                key = torch.roll(key, shifts=(-shift_height, -shift_width), dims=(1, 2))
                value = torch.roll(value, shifts=(-shift_height, -shift_width), dims=(1, 2))
            
            query = rearrange(split_feature(query, num_splits=num_splits, swin_1d=swin_1d), "b h w c -> b (h w) c")
            key = rearrange(split_feature(key, num_splits=num_splits, swin_1d=swin_1d), "b h w c -> b (h w) c")
            value = rearrange(split_feature(value, num_splits=num_splits, swin_1d=swin_1d), "b h w c -> b (h w) c")

            similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale
            if with_shift:
                similarity_matrix += shifted_window_attn_mask.repeat(b * self.num_heads, 1, 1) # [B*N*K*K, H/K*W/K, H/K*W/K].
            
            if self.sink_competition:
                raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
                raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
            else:
                raw_attn = F.softmax(similarity_matrix, dim=-1)
            
            attn = self.dropout(raw_attn)

            out = torch.einsum("bij, bjd -> bid", attn, value)
            out = merge_splits(out.view(b * self.num_heads * num_splits * num_splits, h // num_splits, w // num_splits, c // self.num_heads), None, num_splits, swin_1d=swin_1d)
            if with_shift:
                out = torch.roll(out, shifts=(shift_height, shift_width), dims=(1, 2))
            out = rearrange(out, "(b n) h w c -> b (n c) h w", n=self.num_heads)
            out = to_3d(self.to_out(out), dim=2)
            if not self.pre_norm:
                out = self.norm(out)
            out = to_4d(out, b, h, w, dim=2)

        return out, similarity_matrix


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, receptive_aug: str=None):
        super(TransformerBlock, self).__init__()

        self.cross_attn = AttentionLayer(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            num_heads=num_heads,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.feedforward = FeedForwardLayer(
            in_channels=channels,
            out_channels=channels,
            receptive_aug=receptive_aug,
        )
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor, bi_cross_attn: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if bi_cross_attn:
            cat_feats1 = torch.cat([feats1, feats2], dim=0)
            cat_feats2 = torch.cat([feats2, feats1], dim=0)
            update_cat_feats1, _ = self.cross_attn(cat_feats1.clone(), cat_feats2)
            cat_feats1 = cat_feats1 + update_cat_feats1
            cat_feats1 = cat_feats1 + self.feedforward(cat_feats1.clone())

            feats1, feats2 = cat_feats1.chunk(2, dim=0)

            return feats1, feats2
        else:
            update_feats1, _ = self.cross_attn(feats1.clone(), feats2)
            feats1 = feats1 + update_feats1
            feats1 = feats1 + self.feedforward(feats1.clone())

            return feats1


class SwinTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, receptive_aug: str=None, num_splits: int=1, with_shift: bool=False, swin_1d: bool=False):
        super(SwinTransformerBlock, self).__init__()
        
        self.num_splits = num_splits
        self.with_shift = with_shift
        self.swin_1d = swin_1d

        self.cross_attn = WindowAttentionLayer(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            num_heads=num_heads,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.feedforward = FeedForwardLayer(
            in_channels=channels,
            out_channels=channels,
            receptive_aug=receptive_aug,
        )
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor, bi_cross_attn: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if bi_cross_attn:
            cat_feats1 = torch.cat([feats1, feats2], dim=0)
            cat_feats2 = torch.cat([feats2, feats1], dim=0)
            update_cat_feats1, _ = self.cross_attn(cat_feats1.clone(), cat_feats2, self.num_splits, self.with_shift, self.swin_1d)
            cat_feats1 = cat_feats1 + update_cat_feats1
            cat_feats1 = cat_feats1 + self.feedforward(cat_feats1.clone())

            feats1, feats2 = cat_feats1.chunk(2, dim=0)

            return feats1, feats2
        else:
            update_feats1, _ = self.cross_attn(feats1.clone(), feats2, self.num_splits, self.with_shift, self.swin_1d)
            feats1 = feats1 + update_feats1
            feats1 = feats1 + self.feedforward(feats1.clone())

            return feats1


class VolumeTransformerBlock(nn.Module):
    def __init__(self, cv_channels: int, num_heads: int, expansion: int=2):
        super(VolumeTransformerBlock, self).__init__()

        self.volume_attn = VolumeAttentionLayer(
            in_channels=cv_channels,
            hidden_channels=cv_channels,
            out_channels=cv_channels,
            num_heads=num_heads,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm(cv_channels),
            nn.Linear(cv_channels, expansion * cv_channels),
            nn.SiLU(),
            nn.Linear(expansion * cv_channels, cv_channels),
        )
    
    def forward(self, volume: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor]:
        b, c, d, h, w = volume.shape

        update_volume, attn = self.volume_attn(volume.clone(), feats)
        volume = volume + update_volume
        volume = volume + rearrange(self.feedforward(rearrange(volume, "b c d h w -> (b h w) d c").clone()), "(b h w) d c -> b c d h w", b=b, h=h, w=w)

        return volume, attn


class OutlookTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, receptive_aug: str=None, kernel_size: int=3, padding: int=1, stride: int=1):
        super(OutlookTransformerBlock, self).__init__()

        self.self_attn = OutlookAttentionLayer(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_heads=num_heads,
            dropout=0.0,
            pre_norm=True,
            sink_competition=False,
        )
        self.feedforward = FeedForwardLayer(
            in_channels=channels,
            out_channels=channels,
            receptive_aug=receptive_aug,
        )
    
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        update_feats, _ = self.self_attn(feats.clone())
        feats = feats + update_feats
        feats = feats + self.feedforward(feats.clone())

        return feats


class TransformerFeatureRefiner(nn.Module):
    def __init__(self, channels: int, num_heads: int, depth: int, receptive_aug: str=None):
        super(TransformerFeatureRefiner, self).__init__()
        
        self.depth = depth
        for i in range(depth):
            setattr(
                self,
                f"refinement_{i + 1}",
                TransformerBlock(channels, num_heads, receptive_aug),
            )
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> Tuple[torch.Tensor]:
        for i in range(self.depth):
            feats1, feats2 = getattr(self, f"refinement_{i + 1}")(feats1.float(), feats2.float())

        return feats1, feats2


class SwinTransformerFeatureRefiner(nn.Module):
    def __init__(self, channels: int, num_heads: int, depth: int, receptive_aug: str=None, num_splits: int=4):
        super(SwinTransformerFeatureRefiner, self).__init__()

        self.depth = depth
        for i in range(depth):
            setattr(
                self,
                f"refinement_win_{i + 1}",
                SwinTransformerBlock(channels, num_heads, receptive_aug, num_splits=num_splits, with_shift=False, swin_1d=True)
            )
            setattr(
                self,
                f"refinement_swin_{i + 1}",
                SwinTransformerBlock(channels, num_heads, receptive_aug, num_splits=num_splits, with_shift=True, swin_1d=True)
            )
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> Tuple[torch.Tensor]:
        for i in range(self.depth):
            feats1, feats2 = getattr(self, f"refinement_win_{i + 1}")(feats1.float(), feats2.float())
            feats1, feats2 = getattr(self, f"refinement_swin_{i + 1}")(feats1.float(), feats2.float())
        
        return feats1, feats2
