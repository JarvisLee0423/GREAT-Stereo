from __future__ import division, print_function
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from typing import Union
from utils.utils import autocast, disp_warp, InputPadder
from utils.stereo_matching.data_utils.datasets import *


def count_all_parameters(model: nn.Module) -> Union[int, float]:
    return sum(param.numel() for param in model.parameters())


def count_grad_required_parameters(model: nn.Module) -> Union[int, float]:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def evaluate_eth3d(model: nn.Module, root: str, iters: int=32, mixed_prec: bool=False) -> dict:
    """ Perform evaluation using the ETH3D (train) split. """
    model.eval()
    aug_params = {}
    eval_dataset = ETH3DStereoDataset(aug_params, root)

    out_list, epe_list = [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        (imageL_file, imageR_file, GT_file), left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]
        
        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            outputs = model(left_img, right_img, iters=iters, test_mode=True)
        disp_pr = padder.unpad(outputs[1].float()).cpu().squeeze(0)
        
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.sum((disp_pr - disp_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace("disp0GT.pfm", "mask0nocc.png"))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()
        valid = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)

        out = (epe_flattened > 1.0)
        image_out = out[valid].float().mean().item()
        image_epe = epe_flattened[valid].mean().item()

        logging.info(f"ETH3D {eval_id + 1} out of {len(eval_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}.")
        epe_list.append(image_epe)
        out_list.append(image_out)
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Evaluation ETH3D: EPE {round(epe, 4)}, D1 {round(d1, 4)}.")

    return {"eth3d-epe": epe, "eth3d-d1": d1}


@torch.no_grad()
def evaluate_kitti(model: nn.Module, root: str, iters: int=32, mixed_prec: bool=False) -> dict:
    """ Perform evaluation using the KITTI (train) split. """
    model.eval()
    aug_params = {}
    eval_dataset = KITTIStereoDataset(aug_params, root, image_set="training")
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, occ_out_list, occ_epe_list, nonocc_out_list, nonocc_epe_list, elapsed_list = [], [], [], [], [], [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        _, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]

        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            start = time.time()
            outputs = model(left_img, right_img, iters=iters, test_mode=True)
            end = time.time()
        
        if eval_id > 50:
            elapsed_list.append(end - start)
        disp_pr = padder.unpad(outputs[1]).cpu().squeeze(0)

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.sum((disp_pr - disp_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        valid = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)

        # Compute the epe and d3 for occlusion and non-occlusion parts.
        left_img = padder.unpad(left_img).cpu()
        right_img = padder.unpad(right_img).cpu()
        left_img = 2 * (left_img / 255.0) - 1.0
        right_img = 2 * (right_img / 255.0) - 1.0
        warped_right_img, mask = disp_warp(right_img, disp_gt.unsqueeze(0))
        error = (torch.abs(left_img - warped_right_img).mean(1, keepdim=True) < 0.03).float()
        mask = error * mask
        occ_mask = (1 - mask).flatten().bool() * valid
        nonocc_mask = mask.flatten().bool() * valid

        out = (epe_flattened > 3.0)
        image_out = out[valid].float().mean().item()
        image_epe = epe_flattened[valid].mean().item()
        if eval_id < 9 or (eval_id + 1) % 10 == 0:
            logging.info(f"KITTI Iter {eval_id + 1} out of {len(eval_dataset)}. EPE {round(image_epe, 4)} D3 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS).")
        epe_list.append(epe_flattened[valid].mean().item())
        out_list.append(out[valid].cpu().numpy())
        occ_epe_list.append(epe_flattened[occ_mask].mean().item())
        occ_out_list.append(out[occ_mask].cpu().numpy())
        nonocc_epe_list.append(epe_flattened[nonocc_mask].mean().item())
        nonocc_out_list.append(out[nonocc_mask].cpu().numpy())
    
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    occ_epe_list = np.array(occ_epe_list)
    occ_out_list = np.concatenate(occ_out_list)
    nonocc_epe_list = np.array(nonocc_epe_list)
    nonocc_out_list = np.concatenate(nonocc_out_list)

    epe = np.mean(epe_list)
    d3 = 100 * np.mean(out_list)
    occ_epe = np.mean(occ_epe_list)
    occ_d3 = 100 * np.mean(occ_out_list)
    nonocc_epe = np.mean(nonocc_epe_list)
    nonocc_d3 = 100 * np.mean(nonocc_out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Evaluation KITTI: EPE {round(epe, 4)}, D3 {round(d3, 4)}, Occ-EPE {round(occ_epe, 4)}, Occ-D3 {round(occ_d3, 4)}, Non-Occ-EPE {round(nonocc_epe, 4)}, Non-Occ-D3 {round(nonocc_d3, 4)}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s).")

    return {"kitti-epe": epe, "kitti-d3": d3, "kitti-occ-epe": occ_epe, "kitti-occ-d3": occ_d3, "kitti-nonocc-epe": nonocc_epe, "kitti-nonocc-d3": nonocc_d3}


@torch.no_grad()
def evaluate_sceneflow(model: nn.Module, root: str, iters: int=32, mixed_prec: bool=False) -> dict:
    """ Perform evaluation using the SceneFlow (test) split. """
    model.eval()
    eval_dataset = SceneFlowStereoDataset(root=root, dstype="frames_finalpass", things_test=True)

    out_list, epe_list, occ_out_list, occ_epe_list, nonocc_out_list, nonocc_epe_list = [], [], [], [], [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        _, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]

        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            outputs = model(left_img, right_img, iters=iters, test_mode=True)
        disp_pr = padder.unpad(outputs[1]).cpu().squeeze(0)

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.abs(disp_pr - disp_gt)
        epe = epe.flatten()
        valid = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)

        # Compute the epe and d3 for occlusion and non-occlusion parts.
        left_img = padder.unpad(left_img).cpu()
        right_img = padder.unpad(right_img).cpu()
        left_img = 2 * (left_img / 255.0) - 1.0
        right_img = 2 * (right_img / 255.0) - 1.0
        warped_right_img, mask = disp_warp(right_img, disp_gt.unsqueeze(0))
        error = (torch.abs(left_img - warped_right_img).mean(1, keepdim=True) < 0.03).float()
        mask = error * mask
        occ_mask = (1 - mask).flatten().bool() * valid
        nonocc_mask = mask.flatten().bool() * valid

        if np.isnan(epe[valid].mean().item()):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[valid].mean().item())
        out_list.append(out[valid].cpu().numpy())
        occ_epe_list.append(epe[occ_mask].mean().item())
        occ_out_list.append(out[occ_mask].cpu().numpy())
        nonocc_epe_list.append(epe[nonocc_mask].mean().item())
        nonocc_out_list.append(out[nonocc_mask].cpu().numpy())
    
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    occ_epe_list = np.array(occ_epe_list)
    occ_out_list = np.concatenate(occ_out_list)
    nonocc_epe_list = np.array(nonocc_epe_list)
    nonocc_out_list = np.concatenate(nonocc_out_list)

    epe = np.mean(epe_list)
    d3 = 100 * np.mean(out_list)
    occ_epe = np.mean(occ_epe_list)
    occ_d3 = 100 * np.mean(occ_out_list)
    nonocc_epe = np.mean(nonocc_epe_list)
    nonocc_d3 = 100 * np.mean(nonocc_out_list)

    print(f"Evaluation SceneFlow: EPE {round(epe, 4)}, D3 {round(d3, 4)}, Occ-EPE {round(occ_epe, 4)}, Occ-D3 {round(occ_d3, 4)}, Non-Occ-EPE {round(nonocc_epe, 4)}, Non-Occ-D3 {round(nonocc_d3, 4)}.")

    return {"sceneflow-epe": epe, "sceneflow-d3": d3, "sceneflow-occ-epe": occ_epe, "sceneflow-occ-d3": occ_d3, "sceneflow-nonocc-epe": nonocc_epe, "sceneflow-nonocc-d3": nonocc_d3}


@torch.no_grad()
def evaluate_sceneflow_sequence(model: nn.Module, root: str, iters: int=32, mixed_prec: bool=False, sequence_mode: bool=False) -> dict:
    """ Perform evaluation using the SceneFlow (test) split. """
    model.eval()
    eval_dataset = SceneFlowStereoDataset(root=root, dstype="frames_finalpass", things_test=True)

    out_list, epe_list, occ_out_list, occ_epe_list, nonocc_out_list, nonocc_epe_list = [], [], [], [], [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        _, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]

        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            outputs = model(left_img, right_img, iters=iters, test_mode=True, sequence_mode=sequence_mode)
        disp_prs = outputs[-2][0]

        left_img = padder.unpad(left_img).cpu()
        right_img = padder.unpad(right_img).cpu()
        left_img = 2 * (left_img / 255.0) - 1.0
        right_img = 2 * (right_img / 255.0) - 1.0

        for i, disp_pr in enumerate(disp_prs):
            disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)

            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

            epe = torch.abs(disp_pr - disp_gt)
            epe = epe.flatten()
            valid = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)

            # Compute the epe and d3 for occlusion and non-occlusion parts.
            warped_right_img, mask = disp_warp(right_img, disp_gt.unsqueeze(0))
            error = (torch.abs(left_img - warped_right_img).mean(1, keepdim=True) < 0.03).float()
            mask = error * mask
            occ_mask = (1 - mask).flatten().bool() * valid
            nonocc_mask = mask.flatten().bool() * valid

            if np.isnan(epe[valid].mean().item()):
                continue
            
            out = (epe > 3.0)
            if eval_id == 0:
                epe_list.append([epe[valid].mean().item()])
                out_list.append([out[valid].cpu().numpy()])
                occ_epe_list.append([epe[occ_mask].mean().item()])
                occ_out_list.append([out[occ_mask].cpu().numpy()])
                nonocc_epe_list.append([epe[nonocc_mask].mean().item()])
                nonocc_out_list.append([out[nonocc_mask].cpu().numpy()])
            else:
                epe_list[i].append(epe[valid].mean().item())
                out_list[i].append(out[valid].cpu().numpy())
                occ_epe_list[i].append(epe[occ_mask].mean().item())
                occ_out_list[i].append(out[occ_mask].cpu().numpy())
                nonocc_epe_list[i].append(epe[nonocc_mask].mean().item())
                nonocc_out_list[i].append(out[nonocc_mask].cpu().numpy())
    
    for i in range(len(epe_list)):
        epe_list[i] = np.mean(np.array(epe_list[i]))
        out_list[i] = 100 * np.mean(np.concatenate(out_list[i]))
        occ_epe_list[i] = np.mean(np.array(occ_epe_list[i]))
        occ_out_list[i] = 100 * np.mean(np.concatenate(occ_out_list[i]))
        nonocc_epe_list[i] = np.mean(np.array(nonocc_epe_list[i]))
        nonocc_out_list[i] = 100 * np.mean(np.concatenate(nonocc_out_list[i]))
    
    results = []
    for i in range(len(epe_list)):
        results.append({
            "sceneflow-epe": epe_list[i],
            "sceneflow-d3": out_list[i],
            "sceneflow-occ-epe": occ_epe_list[i],
            "sceneflow-occ-d3": occ_out_list[i],
            "sceneflow-nonocc-epe": nonocc_epe_list[i],
            "sceneflow-nonocc-d3": nonocc_out_list[i]
        })

    print(f"Evaluation SceneFlow: EPE {round(epe_list[-1], 4)}, D3 {round(out_list[-1], 4)}, Occ-EPE {round(occ_epe_list[-1], 4)}, Occ-D3 {round(occ_out_list[-1], 4)}, Non-Occ-EPE {round(nonocc_epe_list[-1], 4)}, Non-Occ-D3 {round(nonocc_out_list[-1], 4)}.")

    return results


# @torch.no_grad()
# def evaluate_sceneflow(model: nn.Module, root: str, iters: int=32, mixed_prec: bool=False) -> dict:
#     """ Peform validation using the Scene Flow (TEST) split """
#     model.eval()
#     eval_dataset = SceneFlowStereoDataset(root=root, dstype="frames_finalpass", things_test=True)
#     eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=8, drop_last=False)

#     out_list, epe_list, rmae_list = [], [], []
#     for i_batch, (_, *data_blob) in enumerate(tqdm(eval_loader)):
#         image1, image2, flow_gt, valid_gt = [x.cuda() for x in data_blob]

#         padder = InputPadder(image1.shape, divis_by=32)
#         image1, image2 = padder.pad(image1, image2)
        
#         with autocast(enabled=mixed_prec):
#             outputs = model(image1, image2, iters=iters, test_mode=True)
#         flow_pr = padder.unpad(outputs[1])
#         assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

#         epe = torch.abs(flow_pr - flow_gt)

#         epe = epe.flatten()
#         val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

#         out = (epe > 1.0).float()
#         epe_list.append(epe[val].mean().item())
#         out_list.append(out[val].mean().item())

#     epe_list = np.array(epe_list)
#     out_list = np.array(out_list)

#     epe = np.mean(epe_list)
#     d1 = 100 * np.mean(out_list)

#     print(f"Evaluation SceneFlow: EPE {round(epe, 4)}, D1 {round(d1, 4)}.")

#     return {"sceneflow-epe": epe, "sceneflow-d1": d1}


@torch.no_grad()
def evaluate_middlebury(model: nn.Module, root: str, iters: int=32, split: str="F", mixed_prec: bool=False) -> dict:
    """ Perform evaluation using the Middlebury dataset. """
    model.eval()
    aug_params = {}
    eval_dataset = MiddleburyStereoDataset(aug_params, root, split="MiddEval3", resolution=split)

    out_list, epe_list = [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        (_, _, disp_file), left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]
        
        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            outputs = model(left_img, right_img, iters=iters, test_mode=True)
        disp_pr = padder.unpad(outputs[1]).cpu().squeeze(0)

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.sum((disp_pr - disp_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(disp_file.replace("disp0GT.pfm", "mask0nocc.png")).convert("L")
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        valid = (valid_gt.reshape(-1) >= 0.5) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[valid].float().mean().item()
        image_epe = epe_flattened[valid].mean().item()
        logging.info(f"Middlebury Iter {eval_id + 1} out of {len(eval_dataset)}. EPE {round(image_epe, 4)} D2 {round(image_out, 4)}.")
        epe_list.append(image_epe)
        out_list.append(image_out)
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d2 = 100 * np.mean(out_list)

    print(f"Evaluation Middlebury-{split}: EPE {round(epe, 4)}, D2 {round(d2, 4)}.")

    return {f"middlebury-{split}-epe": epe, f"middlebury-{split}-d2": d2}


@torch.no_grad()
def evaluate_booster(model: nn.Module, root: str, iters: int=32, split: str="balanced", mixed_prec: bool=False) -> dict:
    """ Perform evaluation using the Booster dataset. """
    model.eval()
    aug_params = {}
    eval_dataset = BoosterStereoDataset(aug_params, root, split=split)

    out_list, epe_list = [], []
    for eval_id in tqdm(range(len(eval_dataset))):
        _, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]
        
        left_img = left_img[None].cuda()
        right_img = right_img[None].cuda()

        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        with autocast(enabled=mixed_prec):
            outputs = model(left_img, right_img, iters=iters, test_mode=True)
        disp_pr = padder.unpad(outputs[1]).cpu().squeeze(0)

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

        epe = torch.sum((disp_pr - disp_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        valid = valid_gt.reshape(-1) >= 0.5
        out = (epe_flattened > 2.0)
        image_out = out[valid].float().mean().item()
        image_epe = epe_flattened[valid].mean().item()
        epe_list.append(image_epe)
        out_list.append(image_out)
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d2 = 100 * np.mean(out_list)

    print(f"Evaluation Booster-{split}: EPE {round(epe, 4)}, D2 {round(d2, 4)}.")

    return {f"booster-{split}-epe": epe, f"booster-{split}-d2": d2}
