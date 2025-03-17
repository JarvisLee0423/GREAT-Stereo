import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Union


def pool2x(inputs: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(inputs, 3, stride=2, padding=1)


def pool4x(inputs: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(inputs, 5, stride=4, padding=1)


def interp(inputs: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    interp_args = {"mode": "bilinear", "align_corners": True}
    return F.interpolate(inputs, dest.shape[2:], **interp_args)


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

        cv_channels = args.cv_levels * (2 * args.cv_radius + 1) * (8 + 1)

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
            cv_channels = args.cv_levels * (2 * args.cv_radius + 1)
        else:
            cv_channels = args.cv_levels * (2 * args.cv_radius + 1) * (8 + 1)

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
