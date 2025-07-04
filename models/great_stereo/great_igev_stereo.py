import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from utils.utils import autocast
from models.great_stereo.updaters import BasicMultiUpdateBlock
from models.great_stereo.cost_volumes import ExcitiveAttentionVolume, CombinedVolumeSampler
from models.great_stereo.basic_modules import context_upsample, Conv2x, Conv2xIN, BasicConvIN
from models.great_stereo.feature_extractors import Mobilenetv2Encoder, MultiBasicEncoder


class GREATStereo(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(GREATStereo, self).__init__()

        self.args = args

        context_channels = args.channels

        self.cnet = MultiBasicEncoder(
            out_channels=[args.channels, context_channels],
            norm_fn="batch",
            downsample=args.n_downsample,
        )

        self.update_block = BasicMultiUpdateBlock(self.args, channels=args.channels)

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_channels[i], args.channels[i] * 3, 3, padding=3 // 2)
            for i in range(self.args.n_gru_layers)
        ])

        self.feature = Mobilenetv2Encoder()

        # Create feature extractor stems.
        self.stem_2 = nn.Sequential(
            BasicConvIN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConvIN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2xIN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConvIN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )

        self.excitive_attention_volume = ExcitiveAttentionVolume(96, self.args.max_disp, 8)
    
    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4, spx_pred).unsqueeze(1)
        
        return up_disp
    
    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor, iters: int=12, disp_init: torch.Tensor=None, test_mode: bool=False) -> Tuple[Union[torch.Tensor, Union[List[torch.Tensor], dict, int, float]]]:
        """ Estimate disparity between pair of frames. """

        left_img = (2 * (left_img / 255.0) - 1.0).contiguous()
        right_img = (2 * (right_img / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            feat_left = self.feature(left_img)
            feat_right = self.feature(right_img)
            stem_2x = self.stem_2(left_img)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(right_img)
            stem_4y = self.stem_4(stem_2y)
            feat_left[0] = torch.cat((feat_left[0], stem_4x), 1)
            feat_right[0] = torch.cat((feat_right[0], stem_4y), 1)

            match_left, match_right, global_volume, init_disp = self.excitive_attention_volume(feat_left, feat_right)

            if not test_mode:
                xspx = self.spx_4(feat_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)
            
            cnet_list = self.cnet(left_img, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
        
        cv_block = CombinedVolumeSampler
        feat_batch, _, feat_h, feat_w = match_left.shape
        cv_fn = cv_block(match_left.float(), match_right.float(), global_volume.float(), radius=self.args.cv_radius, num_levels=self.args.cv_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity.
        for iter in range(iters):
            disp = disp.detach()
            geo_feat = cv_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU.
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru: # Update low-res ConvGRU and mid-res ConvGRU.
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers >= 2)
            
            disp = disp + delta_disp
            if test_mode and iter < iters - 1:
                continue

            # Upsample predictions.
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)
        
        if test_mode:
            return init_disp, disp_up, ({"apc": cv_fn.local_volume_pyramid, "ccv": cv_fn.global_volume_pyramid}, feat_batch, feat_h, feat_w)
        
        init_disp = context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)

        return init_disp, disp_preds
