import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from utils.utils import bilinear_sampler
from models.greaten_stereo.basic_modules import disparity_regression, BasicConv, BasicConvIN
from models.greaten_stereo.feature_extractors import FeatureAttn, AdaptiveGlobalRefiner, HourglassEncoder, GlobalAttentionFeatureRefiner, ChannelExtensionGlobalAttentionFeatureRefiner, ExcitiveAttentionHourglassEncoder, ChannelExtensionExcitiveAttentionHourglassEncoder, ChannelExtensionSimpleExcitiveAttentionHourglassEncoder
from models.greaten_stereo.positions import PositionalEmbeddingCosine2D
from models.greaten_stereo.transformers import TransformerFeatureRefiner


def corr_cost_volume(left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
    cost_volume = torch.sum((left_feat * right_feat), dim=1, keepdim=True)

    return cost_volume


def norm_cost_volume(left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
    cost_volume = torch.mean(((left_feat / (torch.norm(left_feat, 2, 1, True) + 1e-05)) * (right_feat / (torch.norm(right_feat, 2, 1, True) + 1e-05))), dim=1, keepdim=True)

    return cost_volume


def groupwise_cost_volume(left_feat: torch.Tensor, right_feat: torch.Tensor, num_groups: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost_volume = (left_feat * right_feat).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost_volume.shape == (B, num_groups, H, W)

    return cost_volume


def build_corr_volume(left_feat: torch.Tensor, right_feat: torch.Tensor, max_disp: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    cost_volume = left_feat.new_zeros([B, 1, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, :, i, :, i:] = corr_cost_volume(
                left_feat[:, :, :, i:], right_feat[:, :, :, :-i],
            )
        else:
            cost_volume[:, :, i, :, :] = corr_cost_volume(
                left_feat, right_feat,
            )
    cost_volume = cost_volume.contiguous()

    return cost_volume


def build_concat_volume(left_feat: torch.Tensor, right_feat: torch.Tensor, max_disp: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    cost_volume = left_feat.new_zeros([B, 2 * C, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, :C, i, :, :] = left_feat[:, :, :, :]
            cost_volume[:, C:, i, :, i:] = right_feat[:, :, :, :-i]
        else:
            cost_volume[:, :C, i, :, :] = left_feat
            cost_volume[:, C:, i, :, :] = right_feat
    cost_volume = cost_volume.contiguous()

    return cost_volume


def build_norm_volume(left_feat: torch.Tensor, right_feat: torch.Tensor, max_disp: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    cost_volume = left_feat.new_zeros([B, 1, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, :, i, :, i:] = norm_cost_volume(
                left_feat[:, :, :, i:], right_feat[:, :, :, :-i],
            )
        else:
            cost_volume[:, :, i, :, :] = norm_cost_volume(
                left_feat, right_feat,
            )
    cost_volume = cost_volume.contiguous()

    return cost_volume


def build_gwc_volume(left_feat: torch.Tensor, right_feat: torch.Tensor, max_disp: int, num_groups: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    cost_volume = left_feat.new_zeros([B, num_groups, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, :, i, :, i:] = groupwise_cost_volume(
                left_feat[:, :, :, i:], right_feat[:, :, :, :-i], num_groups,
            )
        else:
            cost_volume[:, :, i, :, :] = groupwise_cost_volume(
                left_feat, right_feat, num_groups,
            )
    cost_volume = cost_volume.contiguous()

    return cost_volume


class GeoEncodingVolume(nn.Module):
    def __init__(self, channels: int, max_disp: int, num_groups: int=8):
        super(GeoEncodingVolume, self).__init__()

        self.max_disp = max_disp
        self.num_groups = num_groups

        self.conv = BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)

        self.feature_pos = PositionalEmbeddingCosine2D(96 // 2, normalize=True)
        self.feature_refiner = TransformerFeatureRefiner(96, 1, 4, receptive_aug="conv")

        self.cv_stem = BasicConv(num_groups, num_groups, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.cv_feature_attn = FeatureAttn(num_groups, channels)
        self.cv_agg = HourglassEncoder(num_groups)
        self.classifier = nn.Conv3d(num_groups, 1, 3, 1, 1, bias=False)
    
    def forward(self, feat_left: List[torch.Tensor], feat_right: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # _, _, _, width = feat_left[0].shape

        match_left = self.desc(self.conv(feat_left[0]))
        match_right = self.desc(self.conv(feat_right[0]))

        match_left = match_left + self.feature_pos(match_left)
        match_right = match_right + self.feature_pos(match_right)
        match_left, match_right = self.feature_refiner(match_left.float(), match_right.float())

        # gwc_volume = build_gwc_volume(match_left, match_right, width, self.num_groups)
        gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)
        gwc_volume = self.cv_stem(gwc_volume)
        gwc_volume = self.cv_feature_attn(gwc_volume, feat_left[0])
        geo_encoding_volume = self.cv_agg(gwc_volume, feat_left)

        # Init disparity from geometry encoding volume.
        init_prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        # init_disp = disparity_regression(init_prob, width)
        init_disp = disparity_regression(init_prob, self.max_disp // 4)

        # Init disparity from local volume.
        local_prob = F.softmax(self.classifier(gwc_volume).squeeze(1), dim=1)
        # local_disp = disparity_regression(local_prob, width)
        local_disp = disparity_regression(local_prob, self.max_disp // 4)

        return match_left, match_right, geo_encoding_volume, init_disp, local_disp


class ExcitiveAttentionVolume(nn.Module):
    def __init__(self, channels: int, max_disp: int, num_groups: int=8):
        super(ExcitiveAttentionVolume, self).__init__()

        self.max_disp = max_disp
        self.num_groups = num_groups

        self.global_spatial_refiner = AdaptiveGlobalRefiner(channels)
        self.matching_pos = PositionalEmbeddingCosine2D(channels // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(channels, 1, 4, receptive_aug="conv")

        self.global_context_refiner = GlobalAttentionFeatureRefiner(num_groups)

        self.cc_stem = nn.Sequential(
            BasicConv(channels * 2, channels, is_3d=True, bn=True, relu=True, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv(channels, channels // 2, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels // 2, num_groups, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
        )
        self.feat_stem = nn.Sequential(
            BasicConv(num_groups, num_groups, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_groups, num_groups, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.fuse_stem = BasicConv(num_groups, num_groups, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        
        self.eav_agg = ExcitiveAttentionHourglassEncoder(num_groups)
    
    def forward(self, feat_left: List[torch.Tensor], feat_right: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # Compute the pre-spatial refine for the context feature with the largest resolution.
        match_left = self.global_spatial_refiner(feat_left[0])
        match_right = self.global_spatial_refiner(feat_right[0])

        # Compute the matching attention for the context feature with the largest resolution.
        match_left = match_left + self.matching_pos(match_left)
        match_right = match_right + self.matching_pos(match_right)
        match_left, match_right = self.global_matching_refiner(match_left.float(), match_right.float())

        # Compute the spatial attention for context features.
        context_left = self.global_context_refiner(feat_left)

        # Compute the concatenated feature cost volume.
        cc_volume = build_concat_volume(match_left, match_right, self.max_disp // 4)
        cc_volume = self.cc_stem(cc_volume)
        feat_volume = self.feat_stem(context_left[0]).unsqueeze(2)
        cfc_volume = self.fuse_stem(feat_volume * cc_volume)

        # Aggregate the concatenated feature cost volume by excitive attention mechanism.
        ea_volume, ea_prob = self.eav_agg(cfc_volume, context_left)

        # Get the initialized disparity.
        init_disp = disparity_regression(ea_prob, self.max_disp // 4)

        return match_left, match_right, ea_volume, init_disp


class ChannelExtensionExcitiveAttentionVolume(nn.Module):
    def __init__(self, channels: int, feat_channels: List[int], max_disp: int, num_groups: int=8):
        super(ChannelExtensionExcitiveAttentionVolume, self).__init__()

        self.max_disp = max_disp
        self.num_groups = num_groups

        self.global_spatial_refiner = AdaptiveGlobalRefiner(channels)
        self.matching_pos = PositionalEmbeddingCosine2D(channels // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(channels, 1, 4, receptive_aug="conv")

        self.global_context_refiner = ChannelExtensionGlobalAttentionFeatureRefiner(feat_channels)

        self.cc_stem = nn.Sequential(
            BasicConv(channels * 2, channels, is_3d=True, bn=True, relu=True, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv(channels, channels // 2, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels // 2, num_groups, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
        )
        self.feat_stem = nn.Sequential(
            BasicConv(feat_channels[0], feat_channels[0] // 2, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(feat_channels[0] // 2, num_groups, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.fuse_stem = BasicConv(num_groups, num_groups, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        
        self.eav_agg = ChannelExtensionExcitiveAttentionHourglassEncoder(num_groups, feat_channels)
    
    def forward(self, feat_left: List[torch.Tensor], feat_right: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # Compute the pre-spatial refine for the context feature with the largest resolution.
        match_left = self.global_spatial_refiner(feat_left[0])
        match_right = self.global_spatial_refiner(feat_right[0])

        # Compute the matching attention for the context feature with the largest resolution.
        match_left = match_left + self.matching_pos(match_left)
        match_right = match_right + self.matching_pos(match_right)
        match_left, match_right = self.global_matching_refiner(match_left.float(), match_right.float())

        # Compute the spatial attention for context features.
        context_left = self.global_context_refiner(feat_left)

        # Compute the concatenated feature cost volume.
        cc_volume = build_concat_volume(match_left, match_right, self.max_disp // 4)
        cc_volume = self.cc_stem(cc_volume)
        feat_volume = self.feat_stem(context_left[0]).unsqueeze(2)
        cfc_volume = self.fuse_stem(feat_volume * cc_volume)

        # Aggregate the concatenated feature cost volume by excitive attention mechanism.
        ea_volume, ea_prob = self.eav_agg(cfc_volume, context_left)

        # Get the initialized disparity.
        init_disp = disparity_regression(ea_prob, self.max_disp // 4)

        return match_left, match_right, ea_volume, init_disp


class ChannelExtensionSimpleExcitiveAttentionVolume(nn.Module):
    def __init__(self, channels: int, feat_channels: List[int], max_disp: int, num_groups: int=8):
        super(ChannelExtensionSimpleExcitiveAttentionVolume, self).__init__()

        self.max_disp = max_disp
        self.num_groups = num_groups

        self.global_spatial_refiner = AdaptiveGlobalRefiner(channels)
        self.matching_pos = PositionalEmbeddingCosine2D(channels // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(channels, 1, 4, receptive_aug="conv")

        self.global_context_refiner = ChannelExtensionGlobalAttentionFeatureRefiner(feat_channels)

        self.cc_stem = nn.Sequential(
            BasicConv(channels * 2, channels, is_3d=True, bn=True, relu=True, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv(channels, channels // 2, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels // 2, num_groups, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
        )
        self.feat_stem = nn.Sequential(
            BasicConv(feat_channels[0], feat_channels[0] // 2, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(feat_channels[0] // 2, num_groups, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.fuse_stem = BasicConv(num_groups, num_groups, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        
        self.eav_agg = ChannelExtensionSimpleExcitiveAttentionHourglassEncoder(num_groups, feat_channels)
    
    def forward(self, feat_left: List[torch.Tensor], feat_right: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # Compute the pre-spatial refine for the context feature with the largest resolution.
        match_left = self.global_spatial_refiner(feat_left[0])
        match_right = self.global_spatial_refiner(feat_right[0])

        # Compute the matching attention for the context feature with the largest resolution.
        match_left = match_left + self.matching_pos(match_left)
        match_right = match_right + self.matching_pos(match_right)
        match_left, match_right = self.global_matching_refiner(match_left.float(), match_right.float())

        # Compute the spatial attention for context features.
        context_left = self.global_context_refiner(feat_left)

        # Compute the concatenated feature cost volume.
        cc_volume = build_concat_volume(match_left, match_right, self.max_disp // 4)
        cc_volume = self.cc_stem(cc_volume)
        feat_volume = self.feat_stem(context_left[0]).unsqueeze(2)
        cfc_volume = self.fuse_stem(feat_volume * cc_volume)

        # Aggregate the concatenated feature cost volume by excitive attention mechanism.
        ea_volume, ea_prob = self.eav_agg(cfc_volume, context_left)

        # Get the initialized disparity.
        init_disp = disparity_regression(ea_prob, self.max_disp // 4)

        return match_left, match_right, ea_volume, init_disp


class ChannelExtensionSimpleExcitiveAttentionCombinedVolume(nn.Module):
    def __init__(self, channels: int, feat_channels: List[int], max_disp: int, num_groups: int=8):
        super(ChannelExtensionSimpleExcitiveAttentionCombinedVolume, self).__init__()

        self.max_disp = max_disp
        self.num_groups = num_groups

        # self.global_spatial_refiner = AdaptiveGlobalRefiner(channels)
        self.conv = BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)
        self.matching_pos = PositionalEmbeddingCosine2D(channels // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(channels, 1, 4)

        self.global_context_refiner = ChannelExtensionGlobalAttentionFeatureRefiner(feat_channels)

        self.cc_stem = nn.Sequential(
            BasicConv(channels * 2, channels, is_3d=True, bn=True, relu=True, kernel_size=1, stride=1, padding=0),
            BasicConv(channels, channels // 2, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(channels // 2, num_groups * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        # self.comb_stem = nn.Sequential(
        #     BasicConv(num_groups * 3, num_groups, is_3d=True, bn=True, relu=True, kernel_size=1, stride=1, padding=0),
        # )
        
        self.eav_agg = ChannelExtensionSimpleExcitiveAttentionHourglassEncoder(num_groups * 3, feat_channels)
    
    def forward(self, feat_left: List[torch.Tensor], feat_right: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # Compute the pre-spatial refine for the context feature with the largest resolution.
        # match_left = self.global_spatial_refiner(feat_left[0])
        # match_right = self.global_spatial_refiner(feat_right[0])
        match_left = self.desc(self.conv(feat_left[0]))
        match_right = self.desc(self.conv(feat_right[0]))

        # Compute the matching attention for the context feature with the largest resolution.
        match_left = match_left + self.matching_pos(match_left)
        match_right = match_right + self.matching_pos(match_right)
        match_left, match_right = self.global_matching_refiner(match_left.float(), match_right.float())

        # Compute the spatial attention for context features.
        context_left = self.global_context_refiner(feat_left)

        # Compute the group-wise correlation cost volume.
        corr_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)

        # Compute the concatenated cost volume.
        cat_volume = build_concat_volume(match_left, match_right, self.max_disp // 4)
        cat_volume = self.cc_stem(cat_volume)

        # Compute the combined cost volume.
        combined_volume = torch.cat([corr_volume, cat_volume], dim=1)
        # combined_volume = self.comb_stem(combined_volume)

        # Aggregate the concatenated feature cost volume by excitive attention mechanism.
        ea_volume, ea_prob = self.eav_agg(combined_volume, context_left)

        # Get the initialized disparity.
        init_disp = disparity_regression(ea_prob, self.max_disp // 4)

        return context_left, match_left, match_right, ea_volume, init_disp


class CostVolumeSampler:
    def __init__(self, left_feat: torch.Tensor, right_feat: torch.Tensor, num_levels: int=4, radius: int=4):
        self.num_levels = num_levels
        self.radius = radius
        self.cost_volume_pyramid = []

        # All pairs correlation.
        cost_volume = CostVolumeSampler.compute_cost_volume(left_feat, right_feat)

        batch, h1, w1, _, w2 = cost_volume.shape
        cost_volume = cost_volume.reshape(batch * h1 * w1, 1, 1, w2)

        self.cost_volume_pyramid.append(cost_volume)
        for i in range(self.num_levels):
            cost_volume = F.avg_pool2d(cost_volume, [1, 2], stride=[1, 2])
            self.cost_volume_pyramid.append(cost_volume)
    
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            cost_volume = self.cost_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(2 * r + 1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            cost_volume = bilinear_sampler(cost_volume, coords_lvl)
            cost_volume = cost_volume.view(batch, h1, w1, -1)
            out_pyramid.append(cost_volume)
        
        out = torch.cat(out_pyramid, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def compute_cost_volume(left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        B, D, H, W1 = left_feat.shape
        _, _, _, W2 = right_feat.shape
        left_feat = left_feat.view(B, D, H, W1)
        right_feat = right_feat.view(B, D, H, W2)
        cost_volume = torch.einsum("aijk,aijh->ajkh", left_feat, right_feat)
        cost_volume = cost_volume.reshape(B, H, W1, 1, W2).contiguous()
        
        return cost_volume / torch.sqrt(torch.tensor(D).float())


class CombinedVolumeSampler:
    def __init__(self, init_left_feat: torch.Tensor, init_right_feat: torch.Tensor, global_volume: torch.Tensor, num_levels: int=2, radius: int=4):
        self.num_levels = num_levels
        self.radius = radius
        self.global_volume_pyramid = []
        self.local_volume_pyramid = []

        # All pairs correlation (local volume).
        local_volume = CombinedVolumeSampler.compute_local_cost_volume(init_left_feat, init_right_feat)

        b, c, d, h, w = global_volume.shape
        b, h, w, _, w2 = local_volume.shape

        global_volume = global_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1, d)
        local_volume = local_volume.reshape(b * h * w, 1, 1, w2)

        self.global_volume_pyramid.append(global_volume)
        self.local_volume_pyramid.append(local_volume)
        for i in range(self.num_levels - 1):
            global_volume = F.avg_pool2d(global_volume, [1, 2], stride=[1, 2])
            self.global_volume_pyramid.append(global_volume)
        for i in range(self.num_levels - 1):
            local_volume = F.avg_pool2d(local_volume, [1, 2], stride=[1, 2])
            self.local_volume_pyramid.append(local_volume)
    
    def __call__(self, disp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            global_volume = self.global_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(disp.device)
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)
            global_volume = bilinear_sampler(global_volume, disp_lvl)
            global_volume = global_volume.view(b, h, w, -1)

            local_volume = self.local_volume_pyramid[i]
            local_x0 = coords.reshape(b * h * w, 1, 1, 1) / 2 ** i - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i + dx
            local_coords_lvl = torch.cat([local_x0, y0], dim=-1)
            local_volume = bilinear_sampler(local_volume, local_coords_lvl)
            local_volume = local_volume.view(b, h, w, -1)

            out_pyramid.append(global_volume)
            out_pyramid.append(local_volume)
        
        out = torch.cat(out_pyramid, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def compute_local_cost_volume(left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        B, D, H, W1 = left_feat.shape
        _, _, _, W2 = right_feat.shape
        left_feat = left_feat.view(B, D, H, W1)
        right_feat = right_feat.view(B, D, H, W2)
        local_cost_volume = torch.einsum("aijk,aijh->ajkh", left_feat, right_feat)
        local_cost_volume = local_cost_volume.reshape(B, H, W1, 1, W2).contiguous()

        return local_cost_volume
