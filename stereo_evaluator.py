from __future__ import division, print_function
import logging
import argparse
import importlib
import torch
import torch.nn as nn
from pathlib import Path
from utils.utils import autocast, InputPadder
from utils.stereo_matching.test_utils.evaluators import *
from utils.stereo_matching.test_utils.cost_volume_visualizer import cv_visualizer
from utils.stereo_matching.test_utils.point_cloud_generator import pc_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="raft-stereo", help="name your experiment.")
    parser.add_argument("--logdir", default="./checkpoints/sceneflow", help="the directory to save logs and checkpoints.")
    parser.add_argument("--dataset", help="dataset for evaluation", default="sceneflow", choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in "FHQ"])
    parser.add_argument("--dataset_root", help="dataset root for evaluation", default="/data/sceneflow/")
    parser.add_argument("--restore_ckpt", default=None, help="load the weights from a specific checkpoint.")
    parser.add_argument("--backbone_type", required=False, help="the type of the backbone used in the network ('MobileNetV2' or 'ResidualNet' for default if not set this argument explicitly).")
    parser.add_argument("--backbone_ckpt", required=False, help="the path of the backbone checkpoint.")
    parser.add_argument("--mixed_precision", default=False, action="store_true", help="use mixed precision.")
    parser.add_argument("--precision_dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="choose mixed precision type: float16, bfloat16 or float32.")
    parser.add_argument("--eval_mode", default="metric", choices=["metric", "sequence", "cvvis", "pcgen"], help="evaluation mode.")
    parser.add_argument("--eval_iters", type=int, default=32, help="number of disparity field updates during forward pass.")

    # Architecture choices.
    parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders.")
    parser.add_argument("--cv_levels", type=int, default=4, help="number of levels in the cost volume pyramid.")
    parser.add_argument("--cv_radius", type=int, default=4, help="width of the cost volume pyramid.")
    parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1 / 2 ^ k).")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently.")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels.")
    parser.add_argument("--channels", nargs="+", type=int, default=[128] * 3, help="hidden state and context channels.")
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
    parser.add_argument("--max_disp", type=int, default=192, help="max disparity.")

    args = parser.parse_args()

    # Get model.
    exp_name = args.name
    assert "-stereo" in exp_name, f"Wrong expriment name {exp_name}. Formate of name argument must be 'xxx-stereo'."
    exp_name = exp_name.split(".")[0]
    exp_name = exp_name.replace("-", "_")
    if len(exp_name.split("_")) == 3:
        model_name = exp_name.split("_")[0].upper() + "Stereo"
        module_name = exp_name.split("_")[0] + "_" + exp_name.split("_")[-1]
    else:
        model_name = exp_name.split("_stereo")[0].upper() + "Stereo"
        module_name = exp_name
    module = getattr(importlib.import_module(f"models.{module_name}.{exp_name}"), model_name)
    model = nn.DataParallel(module(args), device_ids=[0])

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint.")
    
    model.cuda()
    model.eval()

    print(f"The model has {format(count_all_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    if args.eval_mode == "metric":
        if args.dataset == "eth3d":
            results = evaluate_eth3d(model, args.dataset_root, iters=args.eval_iters)
        elif args.dataset == "kitti":
            results = evaluate_kitti(model, args.dataset_root, iters=args.eval_iters)
        elif args.dataset in [f"middlebury_{split}" for split in "FHQ"]:
            results = evaluate_middlebury(model, args.dataset_root, iters=args.eval_iters, split=args.dataset[-1])
        elif args.dataset == "booster":
            results = evaluate_booster(model, args.dataset_root, iters=args.eval_iters)
        elif args.dataset == "sceneflow":
            results = evaluate_sceneflow(model, args.dataset_root, iters=args.eval_iters)
        else:
            raise ValueError(f"Can not find the defined dataset's evaluators for {args.dataset}.")
        with open(args.logdir + f"/test_{args.dataset}.txt", "w") as file:
            line = []
            for key, value in results.items():
                line.append(key + f": {round(value, 4)}")
            file.write(" | ".join(line) + "\n")
    elif args.eval_mode == "sequence":
        if args.dataset == "eth3d":
            results = evaluate_eth3d(model, args.dataset_root, iters=args.eval_iters, sequence_mode=True)
        elif args.dataset == "kitti":
            results = evaluate_kitti(model, args.dataset_root, iters=args.eval_iters, sequence_mode=True)
        elif args.dataset in [f"middlebury_{split}" for split in "FHQ"]:
            results = evaluate_middlebury(model, args.dataset_root, iters=args.eval_iters, split=args.dataset[-1], sequence_mode=True)
        elif args.dataset == "sceneflow":
            results = evaluate_sceneflow_sequence(model, args.dataset_root, iters=args.eval_iters, sequence_mode=True)
        else:
            raise ValueError(f"Can not find the defined dataset's evaluators for {args.dataset}.")
        with open(args.logdir + f"/test_sequence_{args.dataset}.txt", "w") as file:
            for i in range(len(results)):
                line = []
                for key, value in results[i].items():
                    line.append(key + f": {round(value, 4)}")
                file.write(f"Iter {i + 1} => " + " | ".join(line) + "\n")
    elif args.eval_mode == "cvvis":
        if args.dataset == "eth3d":
            eval_dataset = ETH3DStereoDataset({}, args.dataset_root, split="test")
        elif args.dataset == "kitti":
            eval_dataset = KITTIStereoDataset({}, args.dataset_root, image_set="testing")
        elif args.dataset in [f"middlebury_{split}" for split in "FHQ"]:
            eval_dataset = MiddleburyStereoDataset({}, args.dataset_root, split="MiddEval3", resolution=args.dataset[-1], test_set=True)
        elif args.dataset == "sceneflow":
            eval_dataset = SceneFlowStereoDataset(root=args.dataset_root, dstype="frames_finalpass", things_test=True)
        else:
            raise ValueError(f"Can not find the defined dataset's evaluators for {args.dataset}.")

        model = model.module
        # model.cpu()
        model.eval()
        while True:
            try:
                eval_id = eval(input(f"Please input the evaludation id (id in [0, {len(eval_dataset) - 1}]): "))
                _, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]

                left_img = left_img[None].cuda()
                right_img = right_img[None].cuda()
                disp_gt = disp_gt[None]
                valid_gt = valid_gt[None][None]

                padder = InputPadder(left_img.shape, divis_by=32)
                left_img, right_img, disp_gt, valid_gt = padder.pad(left_img, right_img, disp_gt, valid_gt)

                with autocast(enabled=False) and torch.no_grad():
                    outputs = model(left_img, right_img, iters=args.eval_iters, test_mode=True)
                
                cv_visualizer(model, left_img, right_img, disp_gt, outputs, args.dataset)

                continue_flag = eval(input("Please input 0 or 1 to give the continue flag ('0' for stop | '1' for continue): "))
                if continue_flag == 0:
                    break
            except IndexError:
                print("Please input valid evaluation id.")
                continue
        print("Cost Volume Visualization Finished!")
    elif args.eval_mode == "pcgen":
        if args.dataset == "eth3d":
            eval_dataset = ETH3DStereoDataset({}, args.dataset_root, split="test")
        elif args.dataset == "kitti":
            eval_dataset = KITTIStereoDataset({}, args.dataset_root, image_set="testing")
        elif args.dataset in [f"middlebury_{split}" for split in "FHQ"]:
            eval_dataset = MiddleburyStereoDataset({}, args.dataset_root, split="MiddEval3", resolution=args.dataset[-1], test_set=True)
        elif args.dataset == "sceneflow":
            eval_dataset = SceneFlowStereoDataset(root=args.dataset_root, dstype="frames_finalpass", things_test=True)
        else:
            raise ValueError(f"Can not find the defined dataset's evaluators for {args.dataset}.")

        model = model.module
        # model.cpu()
        model.eval()
        while True:
            try:
                eval_id = eval(input(f"Please input the evaludation id (id in [0, {len(eval_dataset) - 1}]): "))
                paths, left_img, right_img, disp_gt, valid_gt = eval_dataset[eval_id]

                left_img = left_img[None].cuda()
                right_img = right_img[None].cuda()
                disp_gt = disp_gt[None]
                valid_gt = valid_gt[None][None]

                padder = InputPadder(left_img.shape, divis_by=32)
                left_img, right_img, disp_gt, valid_gt = padder.pad(left_img, right_img, disp_gt, valid_gt)

                with autocast(enabled=False) and torch.no_grad():
                    outputs = model(left_img, right_img, iters=args.eval_iters, test_mode=True)

                left_img = padder.unpad(left_img.float()).cpu().squeeze(0)
                disp_pr = padder.unpad(outputs[1].float()).cpu().squeeze(0)
                disp_gt = padder.unpad(disp_gt.float()).cpu().squeeze(0)
                valid_gt = padder.unpad(valid_gt.float()).cpu().squeeze(0)

                # Obtain the camera information.
                if args.dataset == "sceneflow":
                    focal = 1050.0
                    baseline = 1.0
                    intrinsic = torch.tensor([
                        [1050.0, 0.0, 479.5],
                        [0.0, 1050.0, 269.5],
                        [0.0, 0.0, 1.0],
                    ])
                elif args.dataset == "kitti":
                    if "colored_0" in paths[0]:
                        intrinsic_path = paths[0].replace("colored_0", "calib").replace("_10.png", ".txt")
                        with open(intrinsic_path, "r") as file:
                            intrinsic_data = file.readlines()
                        intrinsic_data = intrinsic_data[0]
                        intrinsic = torch.tensor([eval(num) for num in intrinsic_data.replace("\n", "").replace("P0: ", "").split(" ")]).reshape(3, 4)[:, :3]
                    else:
                        intrinsic_path = paths[0].replace("data_scene_flow", "data_scene_flow_calib").replace("image_2", "calib_cam_to_cam").replace("_10.png", ".txt")
                        with open(intrinsic_path, "r") as file:
                            intrinsic_data = file.readlines()
                        intrinsic_data = intrinsic_data[9]
                        intrinsic = torch.tensor([eval(num) for num in intrinsic_data.replace("\n", "").replace("P_rect_00: ", "").split(" ")]).reshape(3, 4)[:, :3]
                    disp_gt = disp_pr.detach().clone()
                    valid_gt = torch.ones_like(disp_gt)
                    focal = intrinsic[0][0]
                    baseline = 0.54
                elif args.dataset == "eth3d":
                    disp_gt = disp_pr.detach().clone()
                    valid_gt = torch.ones_like(disp_gt)
                    camera_info_path = paths[0].replace("im0.png", "calib.txt")
                    with open(camera_info_path, "r") as file:
                        camera_info_data = file.readlines()
                    intrinsic = torch.tensor([eval(num) for num in " ".join([line for line in camera_info_data[0].replace("]\n", "").replace("cam0=[", "").split("; ")]).split(" ")]).reshape(3, 3)
                    focal = intrinsic[0][0]
                    baseline = eval(camera_info_data[3].split("baseline=")[-1])
                    doffs = eval(camera_info_data[2].split("doffs=")[-1])
                    disp_gt = disp_gt + doffs
                    disp_pr = disp_pr + doffs
                elif "middlebury" in args.dataset:
                    disp_gt = disp_pr.detach().clone()
                    valid_gt = torch.ones_like(disp_gt)
                    camera_info_path = paths[0].replace("im0.png", "calib.txt")
                    with open(camera_info_path, "r") as file:
                        camera_info_data = file.readlines()
                    intrinsic = torch.tensor([eval(num) for num in " ".join([line for line in camera_info_data[0].replace("]\n", "").replace("cam0=[", "").split("; ")]).split(" ")]).reshape(3, 3)
                    focal = intrinsic[0][0]
                    baseline = eval(camera_info_data[3].split("baseline=")[-1])
                    doffs = eval(camera_info_data[2].split("doffs=")[-1])
                    disp_gt = disp_gt + doffs
                    disp_pr = disp_pr + doffs
                else:
                    raise ValueError(f"Can not find the defined dataset for {args.dataset}.")
                
                camera_info = {"focal": focal, "baseline": baseline, "int": intrinsic}

                pc_generator(left_img, valid_gt, disp_gt, disp_pr, camera_info)

                continue_flag = eval(input("Please input 0 or 1 to give the continue flag ('0' for stop | '1' for continue): "))
                if continue_flag == 0:
                    break
            except IndexError:
                print("Please input valid evaluation id.")
                continue
        print("Point Cloud Generation Finished!")
    else:
        raise ValueError(f"Can not find the defined evaluation mode for {args.eval_mode}. Only support 'metric' and 'cvvis' now!")
