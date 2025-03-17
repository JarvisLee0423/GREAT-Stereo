import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEmbeddingCosine2D(nn.Module):
    """
    Relative cosine embedding 2D, partially inspired by DETR (https://github.com/facebookresearch/detr).
    """
    def __init__(self, num_pos_feats: int=128, temperature: int=10000, normalize: bool=False, scale: float=None):
        super(PositionalEmbeddingCosine2D, self).__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("Normalize should be True if scale is passed.")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        :param feat: input feature.
        :return: positional embedding [N, C, H, W].
        """
        if mask is None:
            mask = torch.zeros(
                (feats.size(0), feats.size(2), feats.size(3)),
                device=feats.device,
                dtype=torch.bool,
            )
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=feats.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4,
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
    
    def __repr__(self, _repr_indent: int=4) -> str:
        head = "Positional Embedding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]

        return "\n".join(lines)
