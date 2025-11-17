from __future__ import division, print_function
import logging
import argparse
import importlib
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from utils.stereo_matching.test_utils.evaluators import *


def setup_dist(local_rank: int, world_size: int, seed: int) -> None:
    # Initialize the environment of the distributed data parallel.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize the logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")


def eval_dist(local_rank: int, world_size: int, args: argparse.Namespace) -> str:
    # Set-up the environment of the distributed data parallel.
    setup_dist(local_rank, world_size, args.seed)
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    # Get model.
    exp_name = args.name
    assert "-stereo" in exp_name, f"Wrong experiment name {exp_name}. Formate of name argument must be 'xxx-stereo'."
    exp_name = exp_name.split(".")[0]
    exp_name = exp_name.replace("-", "_")
    if len(exp_name.split("_")) == 3:
        model_name = exp_name.split("_")[0].upper() + "Stereo"
        module_name = exp_name.split("_")[0] + "_" + exp_name.split("_")[-1]
    else:
        model_name = exp_name.split("_stereo")[0].upper() + "Stereo"
        module_name = exp_name
    module = getattr(importlib.import_module(f"models.{module_name}.{exp_name}"), model_name)
    model = module(args)
    model.mark_module_for_freezing()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=True,
    )

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint in GPUs [{local_rank}]...")
        checkpoint = torch.load(args.restore_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint in GPUs [{local_rank}].")

    if is_main_process:
        print(f"All Parameter Count: {count_all_parameters(model.module)}.")
        print(f"Grad Required Parameter Count: {count_grad_required_parameters(model.module)}.")
    
    # Get validation data.
    if args.dataset == "sceneflow":
        val_dataset = SceneFlowStereoDataset(root=args.dataset_root, dstype="frames_finalpass", things_test=True)
        metric = "d3"
    elif "kitti" in args.dataset:
        val_dataset = KITTIStereoDataset(aug_params={}, root=args.dataset_root, image_set="training")
        metric = "d3"
    elif "middlebury" in args.dataset:
        val_dataset = MiddleburyStereoDataset(aug_params={}, root=args.dataset_root, split="MiddEval3", resolution="Q")
        metric = "d2"
    elif "eth3d" in args.dataset:
        val_dataset = ETH3DStereoDataset(aug_params={}, root=args.dataset_root)
        metric = "d1"
    else:
        assert NotImplementedError("Only sceneflow and kitti dataset can be evaluate during training.")
    
    val_dataset_name = args.dataset
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=args.seed, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8 // world_size if args.dataset == "sceneflow" else 1,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    epe_list, out_list, occ_epe_list, occ_out_list, non_occ_epe_list, non_occ_out_list = [], [], [], [], [], []
    dist.barrier()

    if args.dataset == "eth3d":
        epe, out, occ_epe, occ_out, non_occ_epe, non_occ_out = evaluate_dist_eth3d(model, val_loader, device, iters=args.eval_iters, is_main_process=is_main_process)
    elif args.dataset == "kitti":
        epe, out, occ_epe, occ_out, non_occ_epe, non_occ_out = evaluate_dist_kitti(model, val_loader, device, iters=args.eval_iters, is_main_process=is_main_process)
    elif args.dataset in [f"middlebury_{split}" for split in "FHQ"]:
        epe, out, occ_epe, occ_out, non_occ_epe, non_occ_out = evaluate_dist_middlebury(model, val_loader, device, iters=args.eval_iters, is_main_process=is_main_process)
    elif args.dataset == "sceneflow":
        epe, out, occ_epe, occ_out, non_occ_epe, non_occ_out = evaluate_dist_sceneflow(model, val_loader, device, iters=args.eval_iters, is_main_process=is_main_process)
    else:
        raise ValueError(f"Can not find the defined dataset's evaluators for {args.dataset}.")
    dist.barrier()
    epe_list.extend(epe)
    out_list.extend(out)
    occ_epe_list.extend(occ_epe)
    occ_out_list.extend(occ_out)
    non_occ_epe_list.extend(non_occ_epe)
    non_occ_out_list.extend(non_occ_out)
    dist.barrier()

    epe_tensor = torch.tensor(epe_list, device=device, dtype=torch.float32)
    out_tensor = torch.tensor(out_list, device=device, dtype=torch.float32)
    occ_epe_tensor = torch.tensor(occ_epe_list, device=device, dtype=torch.float32)
    occ_out_tensor = torch.tensor(occ_out_list, device=device, dtype=torch.float32)
    non_occ_epe_tensor = torch.tensor(non_occ_epe_list, device=device, dtype=torch.float32)
    non_occ_out_tensor = torch.tensor(non_occ_out_list, device=device, dtype=torch.float32)

    all_epe = [torch.zeros_like(epe_tensor) for _ in range(world_size)]
    all_out = [torch.zeros_like(out_tensor) for _ in range(world_size)]
    all_occ_epe = [torch.zeros_like(occ_epe_tensor) for _ in range(world_size)]
    all_occ_out = [torch.zeros_like(occ_out_tensor) for _ in range(world_size)]
    all_non_occ_epe = [torch.zeros_like(non_occ_epe_tensor) for _ in range(world_size)]
    all_non_occ_out = [torch.zeros_like(non_occ_out_tensor) for _ in range(world_size)]

    dist.all_gather(all_epe, epe_tensor)
    dist.all_gather(all_out, out_tensor)
    dist.all_gather(all_occ_epe, occ_epe_tensor)
    dist.all_gather(all_occ_out, occ_out_tensor)
    dist.all_gather(all_non_occ_epe, non_occ_epe_tensor)
    dist.all_gather(all_non_occ_out, non_occ_out_tensor)

    if is_main_process:
        all_epe = torch.cat(all_epe)
        all_out = torch.cat(all_out)
        all_occ_epe = torch.cat(all_occ_epe)
        all_occ_out = torch.cat(all_occ_out)
        all_non_occ_epe = torch.cat(all_non_occ_epe)
        all_non_occ_out = torch.cat(all_non_occ_out)
        total_epe = all_epe[~torch.isnan(all_epe)].mean().item()
        total_out = all_out[~torch.isnan(all_out)].mean().item() * 100
        total_occ_epe = all_occ_epe[~torch.isnan(all_occ_epe)].mean().item()
        total_occ_out = all_occ_out[~torch.isnan(all_occ_out)].mean().item() * 100
        total_non_occ_epe = all_non_occ_epe[~torch.isnan(all_non_occ_epe)].mean().item()
        total_non_occ_out = all_non_occ_out[~torch.isnan(all_non_occ_out)].mean().item() * 100
    
        results = {
            f"{val_dataset_name}-epe": total_epe,
            f"{val_dataset_name}-{metric}": total_out,
            f"{val_dataset_name}-occ-epe": total_occ_epe,
            f"{val_dataset_name}-occ-{metric}": total_occ_out,
            f"{val_dataset_name}-nonocc-epe": total_non_occ_epe,
            f"{val_dataset_name}-nonocc-{metric}": total_non_occ_out,
        }

        print(f"Evaluation {val_dataset_name}: EPE {round(total_epe, 4)}, D3 {round(total_out, 4)}, Occ-EPE {round(total_occ_epe, 4)}, Occ-D3 {round(total_occ_out, 4)}, Non-Occ-EPE {round(total_non_occ_epe, 4)}, Non-Occ-D3 {round(total_non_occ_out, 4)}.")

        with open(args.logdir + f"/test_{val_dataset_name}.txt", "a") as file:
            line = []
            for key, value in results.items():
                line.append(key + f": {round(value, 4)}")
            file.write(" | ".join(line) + "\n")
    
    print(f"FINISHED EVALUATING in GPUs [{local_rank}].")
    # Destroy the processes.
    dist.destroy_process_group()

    return None


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
    parser.add_argument("--eval_iters", type=int, default=32, help="number of disparity field updates during forward pass.")
    parser.add_argument("--seed", type=int, default=666, help="random seed for the whole system.")

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

    # Create the log directories.
    if torch.multiprocessing.current_process().name == "MainProcess":
        Path(args.logdir).mkdir(exist_ok=True, parents=True)

    # Obtain the world size (number of GPUs).
    world_size = torch.cuda.device_count()
    if torch.multiprocessing.current_process().name == "MainProcess":
        print(f"Using {world_size} GPUs for Distributed Data Parallel training.")
    
    # Start the Distributed Data Parallel training.
    torch.multiprocessing.spawn(
        eval_dist,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
