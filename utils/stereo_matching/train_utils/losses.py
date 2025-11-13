import torch
import torch.nn.functional as F
from typing import List


def get_mask(disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> torch.Tensor:
    # Exclude invalid pixels and extremely large displacements.
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()

    # Exclude extremely large displacements.
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    return valid


def get_metrics(disp_pred: torch.Tensor, disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> dict:
    valid = get_mask(disp_gt, valid, max_disp)

    epe = torch.sum((disp_pred - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "train/epe": epe.mean().item(),
        "train/1px": (epe < 1).float().mean().item(),
        "train/3px": (epe < 3).float().mean().item(),
        "train/5px": (epe < 5).float().mean().item(),
        "train/bad1": (epe > 1).float().mean().item(),
        "train/bad2": (epe > 2).float().mean().item(),
        "train/bad5": (epe > 5).float().mean().item(),
    }

    return metrics


def smooth_l1_loss(disp_pred: torch.Tensor, disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> torch.Tensor:
    disp_loss = 0.0

    valid = get_mask(disp_gt, valid, max_disp)
    valid = valid.bool() & ~torch.isnan(disp_pred)

    disp_loss += F.smooth_l1_loss(disp_pred[valid], disp_gt[valid], size_average=True)

    return disp_loss


def sequence_loss(disp_preds: List[torch.Tensor], disp_gt: torch.Tensor, valid: torch.Tensor, loss_gamma: float=0.9, max_disp: int=700) -> torch.Tensor:
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    valid = get_mask(disp_gt, valid, max_disp)

    for i in range(n_predictions):
        # assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        # We adjust the loss gamma so it is consistent for any number of iterations.
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()
    
    return disp_loss
