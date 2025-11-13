import numpy as np
import matplotlib
import torch
import torch.nn.functional as F
from scipy import interpolate
from typing import Any, List, Tuple, Union

try:
    autocast = torch.cuda.amp.autocast
except:
    # Dummy autocast for PyTorch < 1.6.
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def updisp8(disp: torch.Tensor, mode: str="bilinear") -> torch.Tensor:
    new_size = (8 * disp.shape[2], 8 * disp.shape[3])
    return 8 * F.interpolate(disp, size=new_size, mode=mode, align_corners=True)


def upflow8(flow: torch.Tensor, mode: str="bilinear") -> torch.Tensor:
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def coords_grid(batch: Union[int, float], ht: Union[int, float], wd: Union[int, float], out_second_channel: bool=False) -> torch.Tensor:
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    if out_second_channel:
        coords = coords[1].float()
    else:
        coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def coords_grid_gaussian(batch: Union[int, float], ht: Union[int, float], wd: Union[int, float], gauss_num: Union[int, float], start_point: Any=None) -> torch.Tensor:
    if start_point is None:
        _, x_coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        # coords = torch.stack(coords[::-1], dim=0).float()
        x_coords = x_coords[None].float()
        return x_coords[None, None].repeat(batch, gauss_num, 1, 1, 1)  # [N, seg, 2, H, W].
    else:
        y_coords, x_coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))  # [[ht, wd], [ht, wd]].
        x_coords = x_coords[None] - torch.tensor(start_point).view(gauss_num, 1, 1)
        x_coords = x_coords[None, :, None].repeat(batch, 1, 1, 1, 1).float()
        # coords = torch.cat([x_coords, y_coords.view(1, 1, 1, ht, wd).repeat(batch, gauss_num, 1, 1, 1)], dim=2)
        return x_coords # [batch, gauss_num, 2, ht, wd].


def gauss_blur(input: torch.Tensor, N: Union[int, float]=5, std: Union[int, float]=1) -> torch.Tensor:
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N // 2, torch.arange(N).float() - N // 2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1, 1, N, N).to(input)
    output = F.conv2d(input.reshape(B * D, 1, H, W), weights, padding=N // 2)
    return output.view(B, D, H, W)


def bilinear_sampler(img: torch.Tensor, coords: torch.Tensor, mode: str="bilinear", mask: bool=False) -> torch.Tensor:
    """ Wrapper for grid_sample, uses pixel coordinates. """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem.
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def normalize_coords(grid: torch.Tensor) -> torch.Tensor:
    """
    Normalize coordinates of image scale to [-1, 1].
    Args:
        grid: [B, 2, H, W].
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1 # x: [-1, 1].
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1 # y: [-1, 1].
    grid = grid.permute((0, 2, 3, 1)) # [B, H, W, 2].

    return grid


def meshgrid(inputs: torch.Tensor, homogeneous: bool=False) -> torch.Tensor:
    """
    Generate meshgrid in image scale.
    Args:
        inputs: [B, _, H, W].
        homogeneous: whether to return homogeneous coordinates.
    Return:
        grid: [B, 2, H, W].
    """
    b, _, h, w = inputs.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(inputs) # [1, H, W].
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(inputs)

    grid = torch.cat((x_range, y_range), dim=0) # [2, H, W], grid[:, i, j] = [j, i].
    grid = grid.unsqueeze(0).expand(b, 2, h, w) # [B, 2, H, W].

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w) # [B, 1, H, W].
        grid = torch.cat((grid, ones), dim=1) # [B, 3, H, W].
        assert grid.size(1) == 3
    
    return grid


def disp_warp(img: torch.Tensor, disp: torch.Tensor, padding_mode: str="border") -> Tuple[torch.Tensor]:
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
    assert disp.min() >= 0

    grid = meshgrid(img) # [B, 2, H, W] in image scale.
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1) # [B, 2, H, W].
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid) # [B, H, W, 2] in [-1, 1].
    warped_img = F.grid_sample(img, sample_grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)

    mask = torch.ones_like(img)[:, :1, :, :]
    valid_mask = F.grid_sample(mask, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1

    return warped_img, valid_mask


def gray_2_colormap_np(img: torch.Tensor, cmap: str="rainbow", max: Union[int, float]=None) -> torch.Tensor:
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2, "The wrong dimension for the img, which must be 2."
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def forward_interpolate(disp: torch.Tensor) -> torch.Tensor:
    disp = disp.detach().cpu().numpy()
    dx, dy = disp[0], disp[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    disp_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )
    disp_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    disp = np.stack([disp_x, disp_y], axis=0)
    return torch.from_numpy(disp).float()


class InputPadder:
    """ Pads images such that dimensions are divisible by 8. """
    def __init__(self, dims: Union[list, tuple], mode: str="sintel", divis_by: int=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self.pad_value = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self.pad_value = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
    
    def pad(self, *inputs: Any) -> List[torch.Tensor]:
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self.pad_value, mode="replicate") for x in inputs]
    
    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self.pad_value[2], ht - self.pad_value[3], self.pad_value[0], wd - self.pad_value[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
