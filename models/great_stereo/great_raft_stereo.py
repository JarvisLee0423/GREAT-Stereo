import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from utils.utils import autocast, upflow8, coords_grid
from models.great_stereo.cost_volumes import CostVolumeSampler
from models.great_stereo.updaters import BasicRAFTMultiUpdateBlock
from models.great_stereo.positions import PositionalEmbeddingCosine2D
from models.great_stereo.transformers import TransformerFeatureRefiner
from models.great_stereo.feature_extractors import ResidualBlock, BasicEncoder, MultiBasicEncoder, AdaptiveGlobalRefiner


class GREATStereo(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(GREATStereo, self).__init__()

        self.args = args

        context_channels = args.channels

        self.cnet = MultiBasicEncoder(
            out_channels=[args.channels, context_channels],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        self.update_block = BasicRAFTMultiUpdateBlock(self.args, channels=args.channels)
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_channels[i], args.channels[i] * 3, 3, padding=3 // 2)
            for i in range(self.args.n_gru_layers)
        ])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(out_channels=96, norm_fn="instance", downsample=args.n_downsample)
        
        self.global_spatial_refiner = AdaptiveGlobalRefiner(96)
        self.matching_pos = PositionalEmbeddingCosine2D(96 // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(96, 1, 4, receptive_aug="conv")
    
    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    def initialize_flow(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0. """

        B, _, H, W = img.shape

        coords0 = coords_grid(B, H, W).to(img.device)
        coords1 = coords_grid(B, H, W).to(img.device)

        return coords0, coords1
    
    def upsample_flow(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination. """
        
        B, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(B, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(B, D, factor * H, factor * W)
    
    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor, iters: int=12, flow_init: torch.Tensor=None, test_mode: bool=False) -> Tuple[Union[torch.Tensor, Union[List[torch.Tensor], dict, int, float]]]:
        """ Estimate optical flow between pair of frames. """

        left_img = (2 * (left_img / 255.0) - 1.0).contiguous()
        right_img = (2 * (right_img / 255.0) - 1.0).contiguous()

        # Run the context network.
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((left_img, right_img), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                left_feat, right_feat = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(left_img, num_layers=self.args.n_gru_layers)
                left_feat, right_feat = self.fnet([left_img, right_img])
            
            # Compute the pre-spatial refine for the context feature with the largest resolution.
            match_left = self.global_spatial_refiner(left_feat)
            match_right = self.global_spatial_refiner(right_feat)

            # Compute the matching attention for the context feature with the largest resolution.
            match_left = match_left + self.matching_pos(match_left)
            match_right = match_right + self.matching_pos(match_right)
            match_left, match_right = self.global_matching_refiner(match_left.float(), match_right.float())
            
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning.
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
        
        cv_block = CostVolumeSampler
        feat_batch, _, feat_h, feat_w = match_left.shape
        match_left, match_right = match_left.float(), match_right.float()
        cv_fn = cv_block(match_left, match_right, radius=self.args.cv_radius, num_levels=self.args.cv_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        flow_predictions = []
        for iter in range(iters):
            coords1 = coords1.detach()
            cv = cv_fn(coords1) # Index correlation volume.
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU.
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru: # Update low-res GRU and mid-res GRU.
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers == 3, iter08=True, iter04=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, cv, flow, iter16=self.args.n_gru_layers == 3, iter08=self.args.n_gru_layers >= 2)
            
            # In stereo mode, project flow onto epipolar.
            delta_flow[:, 1] = 0.0

            # F(t + 1) = F(t) + Delta(t).
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode.
            if test_mode and iter < iters - 1:
                continue

            # Upsample predictions.
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = -flow_up[:, :1]

            flow_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up, ({"apc": cv_fn.cost_volume_pyramid}, feat_batch, feat_h, feat_w)
        
        return flow_predictions
