import timm
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from einops import rearrange
from models.greaten_stereo.basic_modules import *
from models.greaten_stereo.positions import PositionalEmbeddingCosine2D
from models.greaten_stereo.transformers import FeedForwardLayer, WindowAttentionLayer, OutlookAttentionLayer, VolumeTransformerBlock, ChannelExtensionVolumeTransformerBlock


class FeatureAttn(nn.Module):
    def __init__(self, cv_channels: int, feat_channels: int):
        super(FeatureAttn, self).__init__()

        self.feat_attn = nn.Sequential(
            BasicConv(feat_channels, feat_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_channels // 2, cv_channels, 1),
        )
    
    def forward(self, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat_attn = self.feat_attn(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_attn) * cv

        return cv


class SimpleVolumeAttn(nn.Module):
    def __init__(self, cv_channels: int, feat_channels: int):
        super(SimpleVolumeAttn, self).__init__()

        self.volume_attn = nn.Sequential(
            BasicConv(cv_channels, cv_channels, is_3d=True, bn=True, relu=True, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1), dilation=(1, 1, 1)),
            nn.Conv3d(cv_channels, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
        )

        self.feat_attn = nn.Sequential(
            BasicConv(feat_channels, feat_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_channels // 2, cv_channels, 1),
        )
    
    def forward(self, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        cv_attn = F.softmax(self.volume_attn(cv), dim=2) # Shape: [b, cc, d, h, w] -> [b, 1, d, h, w].
        feat_attn = self.feat_attn(feat).unsqueeze(2) # Shape: [b, fc, h, w] -> [b, cc, 1, h, w].
        feat_attn = cv_attn * feat_attn # Shape: [b, cc, d, h, w].

        cv = torch.sigmoid(feat_attn) * cv

        return cv


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int):
        super(SimpleUNet, self).__init__()

        self.conv1a = BasicConvReLU(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConvReLU(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConvReLU(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConvReLU(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2xReLU(128, 96, deconv=True)
        self.deconv3a = Conv2xReLU(96, 64, deconv=True)
        self.deconv2a = Conv2xReLU(64, 48, deconv=True)
        self.deconv1a = Conv2xReLU(48, 32, deconv=True)

        self.conv1b = Conv2xReLU(32, 48)
        self.conv2b = Conv2xReLU(48, 64)
        self.conv3b = Conv2xReLU(64, 96)
        self.conv4b = Conv2xReLU(96, 128)

        self.deconv4b = Conv2xReLU(128, 96, deconv=True)
        self.deconv3b = Conv2xReLU(96, 64, deconv=True)
        self.deconv2b = Conv2xReLU(64, 48, deconv=True)
        self.deconv1b = Conv2xReLU(48, in_channels, deconv=True)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rem0 = inputs
        inputs = self.conv1a(inputs)
        rem1 = inputs
        inputs = self.conv2a(inputs)
        rem2 = inputs
        inputs = self.conv3a(inputs)
        rem3 = inputs
        inputs = self.conv4a(inputs)
        rem4 = inputs

        inputs = self.deconv4a(inputs, rem3)
        rem3 = inputs
        inputs = self.deconv3a(inputs, rem2)
        rem2 = inputs
        inputs = self.deconv2a(inputs, rem1)
        rem1 = inputs
        inputs = self.deconv1a(inputs, rem0)
        rem0 = inputs

        inputs = self.conv1b(inputs, rem1)
        rem1 = inputs
        inputs = self.conv2b(inputs, rem2)
        rem2 = inputs
        inputs = self.conv3b(inputs, rem3)
        rem3 = inputs
        inputs = self.conv4b(inputs, rem4)

        inputs = self.deconv4b(inputs, rem3)
        inputs = self.deconv3b(inputs, rem2)
        inputs = self.deconv2b(inputs, rem1)
        inputs = self.deconv1b(inputs, rem0)

        return inputs


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size: int=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        outs = torch.cat([avg_out, max_out], dim=1)
        outs = self.samconv(outs)

        return self.sigmoid(outs)


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_channels: int, ratio: int=16):
        super(ChannelAttentionEnhancement, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(inputs))
        max_out = self.fc(self.max_pool(inputs))
        out = avg_out + max_out

        return self.sigmoid(out)


class AdaptiveGlobalRefiner(nn.Module):
    def __init__(self, channels: int):
        super(AdaptiveGlobalRefiner, self).__init__()

        # Pre-aggregation for the channels.
        self.pre_channel_agg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        )

        # Attention-mask to fuse single-reception and multi-reception features, like in Selective-Stereo: https://github.com/Windsrain/Selective-Stereo.
        self.sam_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.cam_avgpool = nn.AdaptiveAvgPool2d(1)
        self.cam_maxpool = nn.AdaptiveMaxPool2d(1)
        self.cam_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # small-reception features.
        self.small_reception_conv = BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1)

        # large-reception features.
        self.large_repcetion_conv = BasicConvIN(channels, channels, kernel_size=5, padding=2, stride=1)

        # Final fusion features.
        self.final_fusion = nn.Sequential(
            BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1),
        )
    
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Save the raw features.
        residual_feats = feats.clone()
        
        # Compute the channel pre-aggmentation.
        feats = self.pre_channel_agg(feats)

        # Compute the channel attention enhancement.
        cam_avg_out = self.cam_conv(self.cam_avgpool(feats))
        cam_max_out = self.cam_conv(self.cam_maxpool(feats))
        cam_out = cam_avg_out + cam_max_out
        cam_out = self.sigmoid(cam_out)
        feats = cam_out * feats
        # Compute the spatial attention.
        sam_avg_out = torch.mean(feats, dim=1, keepdim=True)
        sam_max_out, _ = torch.max(feats, dim=1, keepdim=True)
        attn_mask = torch.cat([sam_avg_out, sam_max_out], dim=1)
        attn_mask = self.sam_conv(attn_mask)
        attn_mask = self.sigmoid(attn_mask)

        # Compute the small-reception features.
        small_feats = self.small_reception_conv(feats)

        # Compute the large-reception features.
        large_feats = self.large_repcetion_conv(feats)

        # Compute the final refined features.
        refined_feats = small_feats * attn_mask + large_feats * (1 - attn_mask)
        refined_feats = self.final_fusion(residual_feats + refined_feats)

        return refined_feats    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, norm_fn: str="group", stride: int=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = channels // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            if not (stride == 1 and in_channels == channels):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            if not (stride == 1 and in_channels == channels):
                self.norm3 = nn.BatchNorm2d(channels)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            if not (stride == 1 and in_channels == channels):
                self.norm3 = nn.InstanceNorm2d(channels)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_channels == channels):
                self.norm3 = nn.Sequential()
        else:
            raise ValueError(f"Wrong keyword ({norm_fn}) for normalization function! Only 'group', 'batch', 'instance' and 'none' are acceptable.")
    
        if stride == 1 and in_channels == channels:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride),
                self.norm3,
            )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = inputs
        outs = self.conv1(outs)
        outs = self.norm1(outs)
        outs = self.relu(outs)
        outs = self.conv2(outs)
        outs = self.norm2(outs)
        outs = self.relu(outs)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        
        return self.relu(inputs + outs)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, norm_fn: str="group", stride: int=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(channels // 4, channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = channels // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(channels // 4)
            self.norm2 = nn.BatchNorm2d(channels // 4)
            self.norm3 = nn.BatchNorm2d(channels)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(channels)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(channels // 4)
            self.norm2 = nn.InstanceNorm2d(channels // 4)
            self.norm3 = nn.InstanceNorm2d(channels)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(channels)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()
        else:
            raise ValueError(f"Wrong keyword ({norm_fn}) for normalization function! Only 'group', 'batch', 'instance' and 'none' are acceptable.")

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride),
                self.norm4,
            )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = inputs
        outs = self.relu(self.norm1(self.conv1(outs)))
        outs = self.relu(self.norm2(self.conv2(outs)))
        outs = self.relu(self.norm3(self.conv3(outs)))

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        
        return self.relu(inputs + outs)


class Mobilenetv2Encoder(nn.Module):
    def __init__(self):
        super(Mobilenetv2Encoder, self).__init__()

        pretrained = True
        model = timm.create_model("mobilenetv2_100", pretrained=pretrained, features_only=True)
        layers = [1, 2, 3, 5, 6]
        channels = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2xIN(channels[4], channels[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2xIN(channels[3] * 2, channels[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2xIN(channels[2] * 2, channels[1], deconv=True, concat=True)
        self.conv4 = BasicConvIN(channels[1] * 2, channels[1] * 2, kernel_size=3, stride=1, padding=1)
    
    def weight_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                norm = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / norm))
            elif isinstance(module, nn.Conv3d):
                norm = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / norm))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        outs = self.act1(self.bn1(self.conv_stem(inputs)))
        outs2 = self.block0(outs)
        outs4 = self.block1(outs2)
        outs8 = self.block2(outs4)
        outs16 = self.block3(outs8)
        outs32 = self.block4(outs16)

        outs16 = self.deconv32_16(outs32, outs16)
        outs8 = self.deconv16_8(outs16, outs8)
        outs4 = self.deconv8_4(outs8, outs4)
        outs4 = self.conv4(outs4)

        return [outs4, outs8, outs16, outs32]


class HourglassEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super(HourglassEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_attn_8 = FeatureAttn(in_channels * 2, 64)
        self.feature_attn_16 = FeatureAttn(in_channels * 4, 192)
        self.feature_attn_32 = FeatureAttn(in_channels * 6, 160)
        self.feature_attn_up_16 = FeatureAttn(in_channels * 4, 192)
        self.feature_attn_up_8 = FeatureAttn(in_channels * 2, 64)
    
    def forward(self, inputs: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        conv1 = self.conv1(inputs)
        conv1 = self.feature_attn_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_attn_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_attn_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_attn_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_attn_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class GlobalAttentionFeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, cv_channels: int, num_splits: int, kernel_size: int, stride: int, padding: int):
        super(GlobalAttentionFeatureEncoder, self).__init__()

        self.num_splits = num_splits

        self.channels_agg = nn.Sequential(
            BasicConv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, cv_channels, 1),
        )

        self.feature_pos = PositionalEmbeddingCosine2D(cv_channels // 2, normalize=True)

        self.outlook_self_attn = OutlookAttentionLayer(
            in_channels=cv_channels,
            hidden_channels=cv_channels,
            out_channels=cv_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.outlook_self_ffn = FeedForwardLayer(
            in_channels=cv_channels,
            out_channels=cv_channels,
            receptive_aug="conv",
        )

        self.win_self_attn = WindowAttentionLayer(
            in_channels=cv_channels,
            hidden_channels=cv_channels,
            out_channels=cv_channels,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.win_self_ffn = FeedForwardLayer(
            in_channels=cv_channels,
            out_channels=cv_channels,
            receptive_aug="conv",
        )

        self.swin_self_attn = WindowAttentionLayer(
            in_channels=cv_channels,
            hidden_channels=cv_channels,
            out_channels=cv_channels,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.swin_self_ffn = FeedForwardLayer(
            in_channels=cv_channels,
            out_channels=cv_channels,
            receptive_aug="conv",
        )
    
    def forward(self, left_feat: torch.Tensor) -> torch.Tensor:
        # Go through the channel augmentation.
        left_feat = self.channels_agg(left_feat)

        # Compute the outlook self-attention.
        update_outlook_left_feat, _ = self.outlook_self_attn(left_feat.clone())
        left_feat = left_feat + update_outlook_left_feat
        left_feat = left_feat + self.outlook_self_ffn(left_feat.clone())

        # Add positional embedding.
        left_feat = left_feat + self.feature_pos(left_feat)
        # Compute the swin self-attention.
        update_win_left_feat, _ = self.win_self_attn(left_feat.clone(), left_feat, num_splits=self.num_splits, with_shift=False, swin_1d=False)
        left_feat = left_feat + update_win_left_feat
        left_feat = left_feat + self.win_self_ffn(left_feat.clone())
        update_swin_left_feat, _ = self.swin_self_attn(left_feat.clone(), left_feat, num_splits=self.num_splits, with_shift=True, swin_1d=False)
        left_feat = left_feat + update_swin_left_feat
        left_feat = left_feat + self.swin_self_ffn(left_feat.clone())

        return left_feat


class ChannelExtensionGlobalAttentionFeatureEncoder(nn.Module):
    def __init__(self, feat_channels: int, num_splits: int, kernel_size: int, stride: int, padding: int):
        super(ChannelExtensionGlobalAttentionFeatureEncoder, self).__init__()

        self.num_splits = num_splits

        self.feature_pos = PositionalEmbeddingCosine2D(feat_channels // 2, normalize=True)

        self.outlook_self_attn = OutlookAttentionLayer(
            in_channels=feat_channels,
            hidden_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.outlook_self_ffn = FeedForwardLayer(
            in_channels=feat_channels,
            out_channels=feat_channels,
            receptive_aug=None,
        )

        self.win_self_attn = WindowAttentionLayer(
            in_channels=feat_channels,
            hidden_channels=feat_channels,
            out_channels=feat_channels,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.win_self_ffn = FeedForwardLayer(
            in_channels=feat_channels,
            out_channels=feat_channels,
            receptive_aug=None,
        )

        self.swin_self_attn = WindowAttentionLayer(
            in_channels=feat_channels,
            hidden_channels=feat_channels,
            out_channels=feat_channels,
            num_heads=1,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.swin_self_ffn = FeedForwardLayer(
            in_channels=feat_channels,
            out_channels=feat_channels,
            receptive_aug=None,
        )
    
    def forward(self, left_feat: torch.Tensor) -> torch.Tensor:
        # Compute the outlook self-attention.
        update_outlook_left_feat, _ = self.outlook_self_attn(left_feat.clone())
        left_feat = left_feat + update_outlook_left_feat
        left_feat = left_feat + self.outlook_self_ffn(left_feat.clone())

        # Add positional embedding.
        left_feat = left_feat + self.feature_pos(left_feat)
        # Compute the swin self-attention.
        update_win_left_feat, _ = self.win_self_attn(left_feat.clone(), left_feat, num_splits=self.num_splits, with_shift=False, swin_1d=False)
        left_feat = left_feat + update_win_left_feat
        left_feat = left_feat + self.win_self_ffn(left_feat.clone())
        update_swin_left_feat, _ = self.swin_self_attn(left_feat.clone(), left_feat, num_splits=self.num_splits, with_shift=True, swin_1d=False)
        left_feat = left_feat + update_swin_left_feat
        left_feat = left_feat + self.swin_self_ffn(left_feat.clone())

        return left_feat


class GlobalAttentionFeatureRefiner(nn.Module):
    def __init__(self, cv_channels: int=None):
        super(GlobalAttentionFeatureRefiner, self).__init__()

        self.feature_encoder_4 = GlobalAttentionFeatureEncoder(96, cv_channels, num_splits=8, kernel_size=7, stride=1, padding=3)
        self.feature_encoder_8 = GlobalAttentionFeatureEncoder(64, cv_channels * 2, num_splits=4, kernel_size=5, stride=1, padding=2)
        self.feature_encoder_16 = GlobalAttentionFeatureEncoder(192, cv_channels * 4, num_splits=2, kernel_size=3, stride=1, padding=1)
        self.feature_encoder_32 = GlobalAttentionFeatureEncoder(160, cv_channels * 6, num_splits=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, left_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        left_feats_4 = self.feature_encoder_4(left_feats[0])
        left_feats_8 = self.feature_encoder_8(left_feats[1])
        left_feats_16 = self.feature_encoder_16(left_feats[2])
        left_feats_32 = self.feature_encoder_32(left_feats[3])

        return [left_feats_4, left_feats_8, left_feats_16, left_feats_32]


class ChannelExtensionGlobalAttentionFeatureRefiner(nn.Module):
    def __init__(self, feat_channels: List[int]):
        super(ChannelExtensionGlobalAttentionFeatureRefiner, self).__init__()

        self.feature_encoder_4 = ChannelExtensionGlobalAttentionFeatureEncoder(feat_channels[0], num_splits=8, kernel_size=7, stride=1, padding=3)
        self.feature_encoder_8 = ChannelExtensionGlobalAttentionFeatureEncoder(feat_channels[1], num_splits=4, kernel_size=5, stride=1, padding=2)
        self.feature_encoder_16 = ChannelExtensionGlobalAttentionFeatureEncoder(feat_channels[2], num_splits=2, kernel_size=3, stride=1, padding=1)
        self.feature_encoder_32 = ChannelExtensionGlobalAttentionFeatureEncoder(feat_channels[3], num_splits=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, left_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        left_feats_4 = self.feature_encoder_4(left_feats[0])
        left_feats_8 = self.feature_encoder_8(left_feats[1])
        left_feats_16 = self.feature_encoder_16(left_feats[2])
        left_feats_32 = self.feature_encoder_32(left_feats[3])

        return [left_feats_4, left_feats_8, left_feats_16, left_feats_32]


class ExcitiveAttentionHourglassEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super(ExcitiveAttentionHourglassEncoder, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_attn_4 = VolumeTransformerBlock(in_channels, num_heads=1)
        self.feature_attn_8 = VolumeTransformerBlock(in_channels * 2, num_heads=1)
        self.feature_attn_16 = VolumeTransformerBlock(in_channels * 4, num_heads=1)
        self.feature_attn_32 = VolumeTransformerBlock(in_channels * 6, num_heads=1)
        self.feature_attn_up_16 = VolumeTransformerBlock(in_channels * 4, num_heads=1)
        self.feature_attn_up_8 = VolumeTransformerBlock(in_channels * 2, num_heads=1)
    
    def forward(self, inputs: torch.Tensor, left_feats: List[torch.Tensor]) -> torch.Tensor:
        conv0 = self.conv0(inputs)
        conv0, _ = self.feature_attn_4(conv0, left_feats[0])

        conv1 = self.conv1(conv0)
        conv1, _ = self.feature_attn_8(conv1, left_feats[1])

        conv2 = self.conv2(conv1)
        conv2, _ = self.feature_attn_16(conv2, left_feats[2])

        conv3 = self.conv3(conv2)
        conv3, _ = self.feature_attn_32(conv3, left_feats[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2, _ = self.feature_attn_up_16(conv2, left_feats[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1, _ = self.feature_attn_up_8(conv1, left_feats[1])

        conv = self.conv1_up(conv1)

        b, c, d, h, w = conv.shape
        # Save the fist left features into attributes of the module for cost volume evaluation.
        self.cv_left_feat = left_feats[0]
        conv_prob = torch.einsum(
            "bid, bjd -> bij",
            rearrange(conv, "b c d h w -> (b h w) d c"),
            rearrange(left_feats[0], "b c h w -> (b h w) c").unsqueeze(-2),
        ) * (8 ** -0.5)
        conv_prob = F.softmax(conv_prob, dim=-2) + 1e-6
        conv_prob = conv_prob / torch.sum(conv_prob, dim=(-2,), keepdim=True)
        conv_prob = rearrange(conv_prob, "(b h w) d c -> b c d h w", b=b, h=h, w=w).squeeze(1)

        return conv, conv_prob


class ChannelExtensionExcitiveAttentionHourglassEncoder(nn.Module):
    def __init__(self, in_channels: int, feat_channels: List[int]):
        super(ChannelExtensionExcitiveAttentionHourglassEncoder, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_attn_4 = ChannelExtensionVolumeTransformerBlock(feat_channels[0], in_channels, num_heads=1)
        self.feature_attn_8 = ChannelExtensionVolumeTransformerBlock(feat_channels[1], in_channels * 2, num_heads=1)
        self.feature_attn_16 = ChannelExtensionVolumeTransformerBlock(feat_channels[2], in_channels * 4, num_heads=1)
        self.feature_attn_32 = ChannelExtensionVolumeTransformerBlock(feat_channels[3], in_channels * 6, num_heads=1)
        self.feature_attn_up_16 = ChannelExtensionVolumeTransformerBlock(feat_channels[2], in_channels * 4, num_heads=1)
        self.feature_attn_up_8 = ChannelExtensionVolumeTransformerBlock(feat_channels[1], in_channels * 2, num_heads=1)

        self.final_conv = nn.Sequential(
            BasicConv(feat_channels[0], feat_channels[0] // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_channels[0] // 2, in_channels, 1),
        )
    
    def forward(self, inputs: torch.Tensor, left_feats: List[torch.Tensor]) -> torch.Tensor:
        conv0 = self.conv0(inputs)
        conv0, _ = self.feature_attn_4(conv0, left_feats[0])

        conv1 = self.conv1(conv0)
        conv1, _ = self.feature_attn_8(conv1, left_feats[1])

        conv2 = self.conv2(conv1)
        conv2, _ = self.feature_attn_16(conv2, left_feats[2])

        conv3 = self.conv3(conv2)
        conv3, _ = self.feature_attn_32(conv3, left_feats[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2, _ = self.feature_attn_up_16(conv2, left_feats[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1, _ = self.feature_attn_up_8(conv1, left_feats[1])

        conv = self.conv1_up(conv1)

        final_conv = self.final_conv(left_feats[0])

        b, c, d, h, w = conv.shape
        # Save the fist left features into attributes of the module for cost volume evaluation.
        self.cv_left_feat = final_conv
        conv_prob = torch.einsum(
            "bid, bjd -> bij",
            rearrange(conv, "b c d h w -> (b h w) d c"),
            rearrange(final_conv, "b c h w -> (b h w) c").unsqueeze(-2),
        ) * (8 ** -0.5)
        conv_prob = F.softmax(conv_prob, dim=-2) + 1e-6
        conv_prob = conv_prob / torch.sum(conv_prob, dim=(-2,), keepdim=True)
        conv_prob = rearrange(conv_prob, "(b h w) d c -> b c d h w", b=b, h=h, w=w).squeeze(1)

        return conv, conv_prob


class ChannelExtensionSimpleExcitiveAttentionHourglassEncoder(nn.Module):
    def __init__(self, in_channels: int, feat_channels: List[int]):
        super(ChannelExtensionSimpleExcitiveAttentionHourglassEncoder, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_attn_4 = SimpleVolumeAttn(in_channels, feat_channels[0])
        self.feature_attn_8 = SimpleVolumeAttn(in_channels * 2, feat_channels[1])
        self.feature_attn_16 = SimpleVolumeAttn(in_channels * 4, feat_channels[2])
        self.feature_attn_32 = SimpleVolumeAttn(in_channels * 6, feat_channels[3])
        self.feature_attn_up_16 = SimpleVolumeAttn(in_channels * 4, feat_channels[2])
        self.feature_attn_up_8 = SimpleVolumeAttn(in_channels * 2, feat_channels[1])

        self.classifier = nn.Conv3d(in_channels, 1, 3, 1, 1, bias=False)
    
    def forward(self, inputs: torch.Tensor, left_feats: List[torch.Tensor]) -> torch.Tensor:
        conv0 = self.conv0(inputs)
        conv0 = self.feature_attn_4(conv0, left_feats[0])

        conv1 = self.conv1(conv0)
        conv1 = self.feature_attn_8(conv1, left_feats[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_attn_16(conv2, left_feats[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_attn_32(conv3, left_feats[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_attn_up_16(conv2, left_feats[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_attn_up_8(conv1, left_feats[1])

        conv = self.conv1_up(conv1)

        conv_prob = F.softmax(self.classifier(conv).squeeze(1), dim=1)

        return conv, conv_prob


class BasicEncoder(nn.Module):
    def __init__(self, out_channels: int=128, norm_fn: str="batch", dropout: float=0.0, downsample: int=3, pyramid_mode: bool=False):
        super(BasicEncoder, self).__init__()

        self.norm_fn = norm_fn
        self.downsample = downsample
        self.pyramid_mode = pyramid_mode
        channels = [16, 24, 32, 96, 160]

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Wrong keyword ({self.norm_fn}) for normalization function! Only 'group', 'batch', 'instance' and 'none' are acceptable.")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_channels = 64
        if self.pyramid_mode:
            self.layer1 = self.make_layer(channels[2], stride=1)
            self.layer2 = self.make_layer(channels[0], stride=1 + (downsample > 1))
            self.layer3 = self.make_layer(channels[1], stride=1 + (downsample > 1))
            self.layer4 = self.make_layer(channels[2], stride=1 + (downsample > 1))
            self.layer5 = self.make_layer(channels[3], stride=1 + (downsample > 0))
            self.layer6 = self.make_layer(channels[4], stride=1 + (downsample > 0))

            self.deconv32_16 = Conv2xIN(channels[4], channels[3], deconv=True, concat=True)
            self.deconv16_8 = Conv2xIN(channels[3] * 2, channels[2], deconv=True, concat=True)
            self.deconv8_4 = Conv2xIN(channels[2] * 2, channels[1], deconv=True, concat=True)
            self.conv4 = BasicConvIN(channels[1] * 2, channels[1] * 2, kernel_size=3, stride=1, padding=1)
        else:
            self.layer1 = self.make_layer(64, stride=1)
            self.layer2 = self.make_layer(96, stride=1 + (downsample > 1))
            self.layer3 = self.make_layer(128, stride=1 + (downsample > 0))

            # Output convolution.
            self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def make_layer(self, channels: int, stride: int=1) -> nn.Sequential:
        layer1 = ResidualBlock(self.in_channels, channels, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(channels, channels, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_channels = channels

        return nn.Sequential(*layers)
    
    def forward(self, inputs: Union[List[torch.Tensor], Tuple[torch.Tensor], torch.Tensor], dual_inp: bool=False) -> Union[List[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
        # If inputs is tuple or list, combine batch dimension.
        is_list = isinstance(inputs, tuple) or isinstance(inputs, list)
        if is_list:
            batch_dim = inputs[0].shape[0]
            inputs = torch.cat(inputs, dim=0)
        
        outs = self.conv1(inputs)
        outs = self.norm1(outs)
        outs = self.relu1(outs)

        if self.pyramid_mode:
            outs = self.layer1(outs)
            outs2 = self.layer2(outs)
            outs4 = self.layer3(outs2)
            outs8 = self.layer4(outs4)
            outs16 = self.layer5(outs8)
            outs32 = self.layer6(outs16)

            outs16 = self.deconv32_16(outs32, outs16)
            outs8 = self.deconv16_8(outs16, outs8)
            outs4 = self.deconv8_4(outs8, outs4)
            outs4 = self.conv4(outs4)

            if self.training and self.dropout is not None:
                outs4 = self.dropout(outs4)
            
            return [outs4, outs8, outs16, outs32]
        else:
            outs = self.layer1(outs)
            outs = self.layer2(outs)
            outs = self.layer3(outs)

            outs = self.conv2(outs)

            if self.training and self.dropout is not None:
                outs = self.dropout(outs)
            
            if is_list:
                outs = outs.split(split_size=batch_dim, dim=0)
        
            return outs


class MultiBasicEncoder(nn.Module):
    def __init__(self, out_channels: List[Union[int, float]]=[128], norm_fn: str="batch", dropout: float=0.0, downsample: int=3):
        super(MultiBasicEncoder, self).__init__()

        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Wrong keyword ({norm_fn}) for normalization function! Only 'group', 'batch', 'instance' and 'none' are acceptable.")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_channels = 64
        self.layer1 = self.make_layer(64, stride=1)
        self.layer2 = self.make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self.make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self.make_layer(128, stride=2)
        self.layer5 = self.make_layer(128, stride=2)

        output_list = []
        for channel in out_channels:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, channel[2], 3, padding=1),
                nn.InstanceNorm2d(channel[2]) if self.norm_fn == "instance" else nn.Identity(),
                nn.ReLU() if self.norm_fn == "instance" else nn.Identity(),
            )
            output_list.append(conv_out)
        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for channel in out_channels:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, channel[1], 3, padding=1),
                nn.InstanceNorm2d(channel[1]) if self.norm_fn == "instance" else nn.Identity(),
                nn.ReLU() if self.norm_fn == "instance" else nn.Identity(),
            )
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for channel in out_channels:
            conv_out = nn.Sequential(
                nn.Conv2d(128, channel[0], 3, padding=1),
                nn.InstanceNorm2d(channel[0]) if self.norm_fn == "instance" else nn.Identity(),
                nn.ReLU() if self.norm_fn == "instance" else nn.Identity(),
            )
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def make_layer(self, channels: int, stride: int=1) -> nn.Sequential:
        layer1 = ResidualBlock(self.in_channels, channels, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(channels, channels, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_channels = channels

        return nn.Sequential(*layers)
    
    def forward(self, inputs: torch.Tensor, dual_inp: bool=False, num_layers: int=3) -> Tuple[torch.Tensor]:
        outs = self.conv1(inputs)
        outs = self.norm1(outs)
        outs = self.relu1(outs)
        outs = self.layer1(outs)
        outs = self.layer2(outs)
        outs = self.layer3(outs)
        if dual_inp:
            value = outs
            outs = outs[:(outs.shape[0] // 2)]
        
        outputs04 = [func(outs) for func in self.outputs04]
        if num_layers == 1:
            return (outputs04, value) if dual_inp else (outputs04,)
        
        outs = self.layer4(outs)
        outputs08 = [func(outs) for func in self.outputs08]
        if num_layers == 2:
            return (outputs04, outputs08, value) if dual_inp else (outputs04, outputs08,)
        
        outs = self.layer5(outs)
        outputs16 = [func(outs) for func in self.outputs16]

        return (outputs04, outputs08, outputs16, value) if dual_inp else (outputs04, outputs08, outputs16)
