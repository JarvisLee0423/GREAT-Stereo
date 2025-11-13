import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from utils.utils import autocast
from models.greaten_stereo.updaters import BasicMultiUpdateBlock
from models.greaten_stereo.cost_volumes import ChannelExtensionSimpleExcitiveAttentionCombinedVolume, CombinedVolumeSampler
from models.greaten_stereo.basic_modules import context_upsample, Conv2x, Conv2xIN, BasicConvIN
from modules.necks.depth_anything.feat_neck import FeatNeck, ContextNeck
from modules.backbones.depth_anything.depth_anything import DepthAnythingV2, DepthAnythingV2Decoder


class GREATENStereo(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(GREATENStereo, self).__init__()

        self.args = args
        self.freezing_module_list = []

        feat_channels = [96, 64, 192, 160]
        context_channels = args.channels

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        mono_model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        dim_list_config = mono_model_configs[self.args.backbone_type]["features"]
        dim_list = []
        dim_list.append(dim_list_config)

        self.update_block = BasicMultiUpdateBlock(self.args, channels=args.channels)

        # self.context_stem = nn.Sequential(
        #     nn.Conv2d(feat_channels[0], dim_list[0], kernel_size=1, padding=0, stride=1),
        #     nn.Conv2d(feat_channels[1], dim_list[0], kernel_size=1, padding=0, stride=1),
        #     nn.Conv2d(feat_channels[2], dim_list[0], kernel_size=1, padding=0, stride=1),
        # )

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_channels[i], args.channels[i] * 3, 3, padding=3 // 2)
            for i in range(self.args.n_gru_layers)
        ])

        self.feat_neck = FeatNeck(dim_list)
        self.context_neck = ContextNeck(dim_list, output_dim=args.channels[0])
        self.context_fuse = nn.ModuleList([
            nn.Sequential(
                BasicConvIN(dim_list[0] + feat_channels[0], dim_list[0], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(dim_list[0], dim_list[0], kernel_size=1, padding=0, stride=1, bias=False),
            ),
            nn.Sequential(
                BasicConvIN(dim_list[0] + feat_channels[1], dim_list[0], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(dim_list[0], dim_list[0], kernel_size=1, padding=0, stride=1, bias=False),
            ),
            nn.Sequential(
                BasicConvIN(dim_list[0] + feat_channels[2], dim_list[0], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(dim_list[0], dim_list[0], kernel_size=1, padding=0, stride=1, bias=False),
            )
        ])

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
        self.stem_8 = nn.Sequential(
            BasicConvIN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96),
            nn.ReLU(),
        )
        self.stem_16 = nn.Sequential(
            BasicConvIN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192),
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

        self.excitive_attention_volume = ChannelExtensionSimpleExcitiveAttentionCombinedVolume(96, feat_channels, self.args.max_disp, 8)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.backbone_type])
        depth_anything_decoder = DepthAnythingV2Decoder(**mono_model_configs[args.backbone_type])
        state_dict_dpt = torch.load(args.backbone_ckpt, map_location="cpu")
        print(f"Loading ckpt for the backbone {os.path.basename(args.backbone_ckpt)}...")
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        print("Done Loading!")
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        print(f"Freezing the gradient in mono encoder...")
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)
        print("Done Freezing!")

        del depth_anything, depth_anything_decoder, state_dict_dpt

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def freeze_bn(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.SyncBatchNorm) and name in self.freezing_module_list:
                module.eval()
    
    def mark_module_for_freezing(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.freezing_module_list.append(name)
    
    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4, spx_pred).unsqueeze(1)
        
        return up_disp
    
    def infer_mono(self, left_img: torch.Tensor, right_img: torch.Tensor) -> Tuple[torch.Tensor, List[Union[list, torch.Tensor]]]:
        height_ori, width_ori = left_img.shape[2:]
        resize_left_img = F.interpolate(left_img, scale_factor=14 / 16, mode="bilinear", align_corners=True)
        resize_right_img = F.interpolate(right_img, scale_factor=14 / 16, mode="bilinear", align_corners=True)

        patch_h, patch_w = resize_left_img.shape[-2] // 14, resize_left_img.shape[-1] // 14
        feat_left_encoder = self.mono_encoder.get_intermediate_layers(resize_left_img, self.intermediate_layer_idx[self.args.backbone_type], return_class_token=True)
        feat_right_encoder = self.mono_encoder.get_intermediate_layers(resize_right_img, self.intermediate_layer_idx[self.args.backbone_type], return_class_token=True)

        depth_mono = self.mono_decoder(feat_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode="bilinear", align_corners=False)
        feat_left_4x, feat_left_8x, feat_left_16x, feat_left_32x = self.feat_decoder(feat_left_encoder, patch_h, patch_w)
        feat_right_4x, feat_right_8x, feat_right_16x, feat_right_32x = self.feat_decoder(feat_right_encoder, patch_h, patch_w)

        return depth_mono, [feat_left_4x, feat_left_8x, feat_left_16x, feat_left_32x], [feat_right_4x, feat_right_8x, feat_right_16x, feat_right_32x]
    
    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor, iters: int=12, disp_init: torch.Tensor=None, test_mode: bool=False) -> Tuple[Union[torch.Tensor, Union[List[torch.Tensor], dict, int, float]]]:
        """ Estimate disparity between pair of frames. """

        left_img = (2 * (left_img / 255.0) - 1.0).contiguous()
        right_img = (2 * (right_img / 255.0) - 1.0).contiguous()

        _, feat_mono_left, feat_mono_right = self.infer_mono(left_img, right_img)

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            feat_left = self.feat_neck(feat_mono_left)
            feat_right = self.feat_neck(feat_mono_right)
            stem_2x = self.stem_2(left_img)
            stem_4x = self.stem_4(stem_2x)
            stem_8x = self.stem_8(stem_4x)
            stem_16x = self.stem_16(stem_8x)
            stem_x_list = [stem_16x, stem_8x, stem_4x]
            stem_2y = self.stem_2(right_img)
            stem_4y = self.stem_4(stem_2y)

            feat_left[0] = torch.cat((feat_left[0], stem_4x), 1)
            feat_right[0] = torch.cat((feat_right[0], stem_4y), 1)
            
            context_left, match_left, match_right, global_volume, init_disp = self.excitive_attention_volume(feat_left, feat_right)
            
            if not test_mode:
                xspx = self.spx_4(feat_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)
            
            # context_left_list = [
            #     mono_left + self.context_stem[i](torch.relu(context_left)) for i, (mono_left, context_left) in enumerate(zip(feat_mono_left[:-1], context_left[:-1]))
            # ]
            feat_context_left = [
                self.context_fuse[i](torch.cat([mono, context], dim=1)) for i, (mono, context) in enumerate(zip(feat_mono_left[:-1], context_left[:-1]))
            ]
            cnet_list = self.context_neck(feat_context_left, stem_x_list)
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
