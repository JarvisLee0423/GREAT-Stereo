import torch
import torch.nn as nn
from typing import List


class ContextNeck(nn.Module):
    def __init__(self, dim_list: List[int], output_dim: int):
        super(ContextNeck, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0] + 192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0] + 96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0] + 48, output_dim, kernel_size=3, padding=1, stride=1)
    
    def forward(self, features: List[torch.Tensor], stem_x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])

        return features_list


class FeatNeck(nn.Module):
    def __init__(self, dim_list: List[int]):
        super(FeatNeck, self).__init__()

        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48 + dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64 + dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192 + dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192),
            nn.ReLU(),
        )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160),
            nn.ReLU(),
        )

        self.conv_up_32x = nn.ConvTranspose2d(
            160, 192,
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=False,
        )
        self.conv_up_16x = nn.ConvTranspose2d(
            192, 64,
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=False,
        )
        self.conv_up_8x = nn.ConvTranspose2d(
            64, 48,
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=False,
        )

        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)

        return features_mono_list
