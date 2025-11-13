import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Union
from utils.utils import autocast, meshgrid, normalize_coords
from models.greaten_stereo.basic_modules import BasicConvReLU
from models.greaten_stereo.feature_extractors import SimpleUNet


def pool2x(inputs: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(inputs, 3, stride=2, padding=1)


def pool4x(inputs: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(inputs, 5, stride=4, padding=1)


def conv2d(in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, dilation: int=1, groups: int=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            groups=groups
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def interp(inputs: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    original_dtype = inputs.dtype
    inputs_fp32 = inputs.float()
    interp_args = {"mode": "bilinear", "align_corners": True}
    with autocast(enabled=False):
        output_fp32 = F.interpolate(inputs_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output


def interp_mono(inputs: torch.Tensor, sample_grid: torch.Tensor, padding_mode: str) -> torch.Tensor:
    original_dtype = inputs.dtype
    inputs_fp32 = inputs.float()
    sample_grid_fp32 = sample_grid.float()
    with autocast(enabled=False):
        output_fp32 = F.grid_sample(inputs_fp32, sample_grid_fp32, mode="bilinear", padding_mode=padding_mode)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    
    return output


def disp_warp_mono(img: torch.Tensor, disp: torch.Tensor, padding_mode: str="border") -> Tuple[torch.Tensor]:
    """
    Warping by disparity.

    Args:
        img: [B, 3, H, W].
        disp: [B, 1, H, W], positive.
        padding_mode: "zeros" or "border".
    Returns:
        warped_img: [B, 3, H, W].
        valid_mask: [B, 3, H, W].
    """

    grid = meshgrid(img) # [B, 2, H, W] in image scale.
    # Note that -disp here.
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1) # [B, 2, H, W].
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid) # [B, H, W, 2] in [-1, 1].
    warped_img = interp_mono(img, sample_grid, padding_mode)

    mask = torch.ones_like(img)
    valid_mask = interp_mono(mask, sample_grid, padding_mode)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1

    return warped_img, valid_mask


class DispHead(nn.Module):
    def __init__(self, in_channels: int=128, channels: int=256, out_channels: int=1):
        super(DispHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(inputs)))


class FlowHead(nn.Module):
    def __init__(self, in_channels: int=128, channels: int=256, out_channels: int=2):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(inputs)))


class VolumeEncoder(nn.Module):
    def __init__(self, cv_channels: int):
        super(VolumeEncoder, self).__init__()
        self.convv1 = nn.Conv2d(cv_channels, 128, 1, padding=0)
        self.convv2 = nn.Conv2d(128, 96, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, cv: torch.Tensor) -> torch.Tensor:
        return self.convv2(self.relu(self.convv1(cv)))


class ConvGRU(nn.Module):
    def __init__(self, channels: int, in_channels: int, kernel_size: int=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
    
    def forward(self, h: torch.Tensor, cz: torch.Tensor, cr: torch.Tensor, cq: torch.Tensor, *input_list: Any) -> torch.Tensor:
        x = torch.cat(input_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q

        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, channels: int=128, in_channels: int=256, kernel_size: int=3):
        super(SelectiveConvGRU, self).__init__()

        self.convz = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(channels + in_channels, channels, kernel_size, padding=kernel_size // 2)
    
    def forward(self, h: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, inputs], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, inputs], dim=1)))

        h = (1 - z) * h + z * q

        return h


class SelectiveGRU(nn.Module):
    def __init__(self, channels: int=128, in_channels: int=256, small_kernel_size: int=1, large_kernel_size: int=3):
        super(SelectiveGRU, self).__init__()

        self.small_gru = SelectiveConvGRU(channels, in_channels, small_kernel_size)
        self.large_gru = SelectiveConvGRU(channels, in_channels, large_kernel_size)
    
    def forward(self, attn: torch.Tensor, h: torch.Tensor, *inputs: Any) -> torch.Tensor:
        inputs = torch.cat(inputs, dim=1)
        h = self.small_gru(h, inputs) * attn + self.large_gru(h, inputs) * (1 - attn)

        return h


class SepConvGRU(nn.Module):
    def __init__(self, channels: int=128, in_channels: int=192 + 128):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(channels + in_channels, channels, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(channels + in_channels, channels, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(channels + in_channels, channels, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(channels + in_channels, channels, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(channels + in_channels, channels, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(channels + in_channels, channels, (5, 1), padding=(2, 0))
    
    def forward(self, h: torch.Tensor, *inputs: Any) -> torch.Tensor:
        # Horizontal.
        inputs = torch.cat(inputs, dim=1)
        hx = torch.cat([h, inputs], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, inputs], dim=1)))
        h = (1 - z) * h + z * q

        # Vertical.
        hx = torch.cat([h, inputs], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, inputs], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(BasicMotionEncoder, self).__init__()

        self.args = args

        cv_channels = args.cv_levels * (2 * args.cv_radius + 1) * (24 + 1)

        self.convc1 = nn.Conv2d(cv_channels, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)
    
    def forward(self, disp: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        cv = F.relu(self.convc1(cv))
        cv = F.relu(self.convc2(cv))
        disp_feat = F.relu(self.convd1(disp))
        disp_feat = F.relu(self.convd2(disp_feat))

        cv_disp = torch.cat([cv, disp_feat], dim=1)
        out = F.relu(self.conv(cv_disp))
        
        return torch.cat([out, disp], dim=1)


class RAFTMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(RAFTMotionEncoder, self).__init__()

        self.args = args

        cv_channels = args.cv_levels * (2 * args.cv_radius + 1)

        self.convc1 = nn.Conv2d(cv_channels, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)
    
    def forward(self, flow: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        cv = F.relu(self.convc1(cv))
        cv = F.relu(self.convc2(cv))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cv_flo = torch.cat([cv, flo], dim=1)
        out = F.relu(self.conv(cv_flo))

        return torch.cat([out, flow], dim=1)


class SelectiveMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(SelectiveMotionEncoder, self).__init__()

        self.args = args

        if "raft" in self.args.name:
            cv_channels = args.cv_levels * (2 * args.cv_radius + 1) * (24 + 1)
        else:
            cv_channels = args.cv_levels * (2 * args.cv_radius + 1) * (24 + 1)

        self.convc1 = nn.Conv2d(cv_channels, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        if "raft" in self.args.name:
            self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
            self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        else:
            self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
            self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)
    
    def forward(self, disp: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        cv = F.relu(self.convc1(cv))
        cv = F.relu(self.convc2(cv))
        if "raft" in self.args.name:
            disp_ = F.relu(self.convf1(disp))
            disp_ = F.relu(self.convf2(disp_))
        else:
            disp_ = F.relu(self.convd1(disp))
            disp_ = F.relu(self.convd2(disp_))

        cv_disp = torch.cat([cv, disp_], dim=1)
        out = F.relu(self.conv(cv_disp))

        return torch.cat([out, disp], dim=1)


class CombinedMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(CombinedMotionEncoder, self).__init__()

        self.args = args

        cv_channels = (2 * args.cv_radius + 1) * 2 + 96

        self.convc1 = nn.Conv2d(cv_channels, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 32, 7, padding=3)
        self.convd2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv = nn.Conv2d(96 + 32, 128 - 1, 3, padding=1)
    
    def forward(self, disp: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        cv = F.relu(self.convc1(cv))
        cv = F.relu(self.convc2(cv))
        disp_feat = F.relu(self.convd1(disp))
        disp_feat = F.relu(self.convd2(disp_feat))

        cv_disp = torch.cat([cv, disp_feat], dim=1)
        out = F.relu(self.conv(cv_disp))
        
        return torch.cat([out, disp], dim=1)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, channels: list=[]):
        super(BasicMultiUpdateBlock, self).__init__()

        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_out_channels = 128

        self.gru04 = ConvGRU(channels[2], encoder_out_channels + channels[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(channels[1], channels[0] * (args.n_gru_layers == 3) + channels[2])
        self.gru16 = ConvGRU(channels[0], channels[1])
        self.disp_head = DispHead(channels[2], channels=256, out_channels=1)
        factor = 2 ** self.args.n_downsample

        if "raft" in self.args.name:
            self.mask = nn.Sequential(
                nn.Conv2d(channels[2], 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0),
            )
        else:
            self.mask_feat_4 = nn.Sequential(
                nn.Conv2d(channels[2], 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, net: List[torch.Tensor], inp: List[List[torch.Tensor]], cv: torch.Tensor=None, disp: torch.Tensor=None, iter04: bool=True, iter08: bool=True, iter16: bool=True, update: bool=True) -> Tuple[torch.Tensor]:
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, cv)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)
        
        if not update:
            return net
        
        delta_disp = self.disp_head(net[0])
        if "raft" in self.args.name:
            mask_feat_4 = 0.25 * self.mask(net[0])
        else:
            mask_feat_4 = self.mask_feat_4(net[0])

        return net, mask_feat_4, delta_disp


class BasicRAFTMultiUpdateBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, channels: list=[]):
        super(BasicRAFTMultiUpdateBlock, self).__init__()

        self.args = args

        self.encoder = RAFTMotionEncoder(args)
        encoder_out_channels = 128

        self.gru04 = ConvGRU(channels[2], encoder_out_channels + channels[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(channels[1], channels[0] * (args.n_gru_layers == 3) + channels[2])
        self.gru16 = ConvGRU(channels[0], channels[1])
        self.flow_head = FlowHead(channels[2], channels=256, out_channels=2)
        factor = 2 ** self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(channels[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0),
        )
    
    def forward(self, net: List[torch.Tensor], inp: List[List[torch.Tensor]], cv: torch.Tensor=None, flow: torch.Tensor=None, iter04: bool=True, iter08: bool=True, iter16: bool=True, update: bool=True) -> Tuple[torch.Tensor]:
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(flow, cv)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)
        
        if not update:
            return net
        
        delta_flow = self.flow_head(net[0])

        # Scale mask to balance gradients.
        mask = 0.25 * self.mask(net[0])

        return net, mask, delta_flow


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, channels: Union[List[int], int]=128):
        super(BasicSelectiveMultiUpdateBlock, self).__init__()

        self.args = args
        self.encoder = SelectiveMotionEncoder(args)
        channels = channels[0] if "raft" in self.args.name else channels
        encoder_out_channels = 128

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveGRU(channels, channels * 2) if "raft" in self.args.name else SelectiveGRU(channels[0], channels[0] + channels[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveGRU(channels, channels * (args.n_gru_layers == 3) + channels * 2) if "raft" in self.args.name else SelectiveGRU(channels[1], channels[0] * (args.n_gru_layers == 3) + channels[1] + channels[2])
        self.gru04 = SelectiveGRU(channels, channels * (args.n_gru_layers > 1) + channels * 2) if "raft" in self.args.name else SelectiveGRU(channels[2], encoder_out_channels + channels[1] * (args.n_gru_layers > 1) + channels[2])

        self.disp_head = DispHead(channels, 256) if "raft" in self.args.name else DispHead(channels[2], 256)
        factor = 2 ** self.args.n_downsample

        if "raft" in self.args.name:
            self.mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0),
            )
        else:
            self.mask_feat_4 = nn.Sequential(
                nn.Conv2d(channels[2], 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, net: List[torch.Tensor], inp: List[torch.Tensor], cv: torch.Tensor, disp: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(attn[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(attn[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(attn[1], net[1], inp[1], pool2x(net[0]))
        
        motion_features = self.encoder(disp, cv)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(attn[0], net[0], motion_features, interp(net[1], net[0]))
        
        delta_disp = self.disp_head(net[0])

        # Scale mask to balance gradients.
        if "raft" in self.args.name:
            mask = 0.25 * self.mask(net[0])
        else:
            mask = 0.25 * self.mask_feat_4(net[0])

        return net, mask, delta_disp


class MonoDepthMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(MonoDepthMotionEncoder, self).__init__()

        self.args = args

        cv_channels = 96 + args.cv_levels * (2 * args.cv_radius + 1) * (24 + 1)

        self.convc1 = nn.Conv2d(cv_channels, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc1_mono = nn.Conv2d(cv_channels, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(128, 64 - 1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64 - 1, 3, padding=1)
    
    def forward(self, disp: torch.Tensor, cv: torch.Tensor, flaw_stereo: torch.Tensor, disp_mono: torch.Tensor, cv_mono: torch.Tensor, flaw_mono: torch.Tensor) -> torch.Tensor:
        cv = F.relu(self.convc1(torch.cat([cv, flaw_stereo], dim=1)))
        cv = F.relu(self.convc2(cv))
        cv_mono = F.relu(self.convc1_mono(torch.cat([cv_mono, flaw_mono], dim=1)))
        cv_mono = F.relu(self.convc2_mono(cv_mono))
        disp_feat = F.relu(self.convd1(disp))
        disp_feat = F.relu(self.convd2(disp_feat))
        disp_mono_feat = F.relu(self.convd1_mono(disp_mono))
        disp_mono_feat = F.relu(self.convd2_mono(disp_mono_feat))

        cv_disp = torch.cat([cv, disp_feat], dim=1)
        cv_disp_mono = torch.cat([cv_mono, disp_mono_feat], dim=1)
        out = F.relu(self.conv(cv_disp))
        out_mono = F.relu(self.conv_mono(cv_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)


class MonoDepthMultiUpdateBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, channels: list=[]):
        super(MonoDepthMultiUpdateBlock, self).__init__()

        self.args = args
        self.encoder = MonoDepthMotionEncoder(args)
        encoder_out_channels = 128

        self.gru04 = ConvGRU(channels[2], encoder_out_channels + channels[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(channels[1], channels[0] * (args.n_gru_layers == 3) + channels[2])
        self.gru16 = ConvGRU(channels[0], channels[1])
        self.disp_head = DispHead(channels[2], channels=256, out_channels=1)
        factor = 2 ** self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(channels[2], 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, net: List[torch.Tensor], inp: List[List[torch.Tensor]], flaw_stereo: torch.Tensor=None, disp: torch.Tensor=None, cv: torch.Tensor=None, flaw_mono: torch.Tensor=None, disp_mono: torch.Tensor=None, cv_mono: torch.Tensor=None, iter04: bool=True, iter08: bool=True, iter16: bool=True, update: bool=True) -> Tuple[torch.Tensor]:
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, cv, flaw_stereo, disp_mono, cv_mono, flaw_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)
        
        if not update:
            return net
        
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])

        return net, mask_feat_4, delta_disp


class REMP(nn.Module):
    """
    Height and width need to be divided by 16.
    """
    def __init__(self):
        super(REMP, self).__init__()

        # Left and warped flaw.
        in_channels = 6
        channel = 32
        self.conv1_mono = conv2d(in_channels, 16)
        self.conv1_stereo = conv2d(in_channels, 16)
        self.conv2_mono = conv2d(1, 16) # On low disparity.
        self.conv2_stereo = conv2d(1, 16) # On low disparity.

        self.conv_start = BasicConvReLU(64, channel, kernel_size=3, padding=2, dilation=2)
        self.refinement_block = SimpleUNet(in_channels=channel)
        self.ap = nn.AdaptiveAvgPool2d(1)
        
        self.lfe = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.lmc = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(channel, channel * 2, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 3, padding=(3 // 2), bias=True),
            nn.Sigmoid(),
        )

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
    
    def forward(self, disp_mono: torch.Tensor, disp_stereo: torch.Tensor, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        assert disp_mono.dim() == 4, "The length of shape of disp_mono is not 4."
        assert disp_stereo.dim() == 4, "The length of shape of disp_stereo is not 4."

        warped_right_mono = disp_warp_mono(right_img, disp_mono)[0] # [B, 3, H, W].
        flaw_mono = warped_right_mono - left_img # [B, 3, H, W].

        warped_right_stereo = disp_warp_mono(right_img, disp_stereo)[0] # [B, 3, H, W].
        flaw_stereo = warped_right_stereo - left_img

        ref_flaw_mono = torch.cat((flaw_mono, left_img), dim=1) # [B, 6, H, W].
        ref_flaw_stereo = torch.cat((flaw_stereo, left_img), dim=1) # [B, 6, H, W].

        ref_flaw_mono = self.conv1_mono(ref_flaw_mono) # [B, 16, H, W].
        ref_flaw_stereo = self.conv1_stereo(ref_flaw_stereo) # [B, 16, H, W].

        disp_feat_mono = self.conv2_mono(disp_mono) # [B, 16, H, W].
        disp_feat_stereo = self.conv2_stereo(disp_stereo) # [B, 16, H, W].

        x = torch.cat((ref_flaw_mono, disp_feat_mono, ref_flaw_stereo, disp_feat_stereo), dim=1) # [B, 64, H, W].
        x = self.conv_start(x) # [B, 32, H, W].
        x = self.refinement_block(x) # [B, 32, H, W].

        low = self.lfe(self.ap(x))
        motif = self.lmc(x)
        x = torch.mul((1 - motif), low) + torch.mul(motif, x)
        x = self.final_conv(x) # [B, 1, H, W].

        disp_stereo = nn.LeakyReLU()(disp_stereo + x) # [B, 1, H, W].

        return disp_stereo
