import torch
from typing import Any


def merge_splits(splits: torch.Tensor, height: int, num_splits: int=2, swin_1d: bool=True) -> torch.Tensor:
    if swin_1d:
        b, w, c = splits.shape
        b_new = b // num_splits // height

        splits = splits.view(b_new, height, num_splits, w, c)
        merge = splits.view(b_new, height, num_splits * w, c)
    else:
        b, h, w, c = splits.shape
        b_new = b // num_splits // num_splits

        splits = splits.view(b_new, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(b_new, num_splits * h, num_splits * w, c)
    
    return merge


def split_feature(feature: torch.Tensor, num_splits: int=2, swin_1d: bool=True) -> torch.Tensor:
    if swin_1d:
        b, w, c = feature.shape
        assert w % num_splits == 0, "Please check the value of num_splits, it has to match the width of the feature."

        b_new = b * num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, w // num_splits, c).view(b_new, w_new, c)
    else:
        b, h, w, c = feature.shape
        assert h % num_splits == 0 and w % num_splits == 0, "Please check the value of num_splits, it has to match to the resolution of the feature."

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)

    return feature


def generate_shift_window_attn_mask(height: int, width: int, window_height: int, window_width: int, shift_height: int, shift_width: int, device: Any, swin_1d: bool=True) -> torch.Tensor:
    if swin_1d:
        img_mask = torch.zeros((1, width, 1)).to(device)
        w_slices = (
            slice(0, -window_width),
            slice(-window_width, -shift_width),
            slice(-shift_width, None),
        )
        cnt = 0
        for w in w_slices:
            img_mask[:, w, :] = cnt
            cnt += 1
        
        mask_windows = split_feature(img_mask, num_splits=width // window_width, swin_1d=swin_1d)
        mask_windows = mask_windows.view(-1, window_width)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        h, w = height, width
        img_mask = torch.zeros((1, h, w, 1)).to(device)
        h_slices = (
            slice(0, -window_height),
            slice(-window_height, -shift_height),
            slice(-shift_height, None),
        )
        w_slices = (
            slice(0, -window_width),
            slice(-window_width, -shift_width),
            slice(-shift_width, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = split_feature(img_mask, num_splits=width // window_width, swin_1d=swin_1d)
        mask_windows = mask_windows.view(-1, window_height * window_width)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask
