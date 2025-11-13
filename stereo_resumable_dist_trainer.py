from __future__ import division, print_function
import os
import logging
import argparse
import importlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pathlib import Path
from utils.stereo_matching.test_utils.evaluators import *
from utils.stereo_matching.train_utils.loggers import Logger
from utils.stereo_matching.train_utils.losses import get_metrics, smooth_l1_loss, sequence_loss
from utils.stereo_matching.data_utils.datasets import *
from utils.stereo_matching.data_utils.data_fetchers import fetch_training_data

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


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


def train_dist(local_rank: int, world_size: int, args: argparse.Namespace) -> str:
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

    if is_main_process:
        print(f"All Parameter Count: {count_all_parameters(model.module)}.")
        print(f"Grad Required Parameter Count: {count_grad_required_parameters(model.module)}.")

        # Save the training settings.
        args_dict = vars(args)
        with open(args.logdir + "/training_settings.txt", "w") as file:
            for key, value in args_dict.items():
                file.write("{:<20} {:<10}".format(key, str(value)) + "\n")
        
        # Save the model settings.
        with open(args.logdir + "/model_settings.txt", "w") as file:
            file.write("All Model Parameters: " + str(count_all_parameters(model.module)) + "\n")
            file.write("Grad Required Model Parameters: " + str(count_grad_required_parameters(model.module)) + "\n")
            file.write(model.module.__str__())
    
    # Get training data.
    train_dataset = fetch_training_data(args, dataset_only=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    # Get validation data.
    for dataset_name, dataset_root in zip(args.train_datasets, args.train_datasets_root):
        if dataset_name == "sceneflow":
            val_dataset = SceneFlowStereoDataset(root=dataset_root, dstype="frames_finalpass", things_test=True)
        elif "kitti" in dataset_name:
            val_dataset = KITTIStereoDataset(aug_params={}, root=f"{dataset_root}/2015/data_scene_flow", image_set="training")
        elif "middlebury" in dataset_name:
            val_dataset = MiddleburyStereoDataset(aug_params={}, root=dataset_root, split="MiddEval3", resolution="H")
        elif "eth3d" in dataset_name:
            val_dataset = ETH3DStereoDataset(aug_params={}, root=dataset_root)
        else:
            assert NotImplementedError("Only sceneflow and kitti dataset can be evaluate during training.")
    val_dataset_name = dataset_name
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=args.seed, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size // world_size if val_dataset_name == "sceneflow" else 1,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
    )

    # Get optimizer and scheduler.
    optim_name = args.optimizer
    fetch_optimizer_func = getattr(importlib.import_module("utils.stereo_matching.train_utils.optimizers"), f"fetch_{optim_name}_optimizer")
    optimizer, scheduler = fetch_optimizer_func(args, model.module)

    # Get logger.
    logger = Logger(args, model.module, scheduler) if is_main_process else None

    # Load the checkpoint, if provide in just the main process.
    restore_total_step = None
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint in GPUs [{local_rank}]...")
        checkpoint = torch.load(args.restore_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint in GPUs [{local_rank}].")
        optimizer_ckpt_path = args.restore_ckpt.replace(os.path.basename(args.restore_ckpt), "optimizer") + f"/{os.path.basename(args.restore_ckpt)}"
        scheduler_ckpt_path = args.restore_ckpt.replace(os.path.basename(args.restore_ckpt), "scheduler") + f"/{os.path.basename(args.restore_ckpt)}"
        if os.path.exists(optimizer_ckpt_path) and os.path.exists(scheduler_ckpt_path):
            logging.info(f"Loading optimizer in GPUs [{local_rank}]...")
            optimizer_ckpt = torch.load(optimizer_ckpt_path, map_location=device)
            optimizer.load_state_dict(optimizer_ckpt)
            logging.info(f"Done loading optimizer in GPUs [{local_rank}].")
            logging.info(f"Loading scheduler in GPUs [{local_rank}]...")
            scheduler_ckpt = torch.load(scheduler_ckpt_path, map_location=device)
            scheduler.load_state_dict(scheduler_ckpt)
            logging.info(f"Done loading scheduler in GPUs [{local_rank}].")
            restore_total_step = eval(os.path.basename(args.restore_ckpt).split("_")[0])
            if is_main_process and logger is not None:
                logger.total_steps = restore_total_step
    
    # Ensure the synchronization.
    if args.restore_ckpt is not None:
        dist.barrier()
    
    # Start training the model.
    model.train()
    # We keep BatchNorm2D frozen.
    model.module.freeze_bn()
    # Check whether the freezing pipeline is correct.
    for name, module in model.module.named_modules():
        if isinstance(module, nn.SyncBatchNorm):
            if name in model.module.freezing_module_list:
                print(f"RANK {local_rank}: nn.BatchNorm2d -> nn.SyncBatchNorm ({name}): train mode is {module.training}.")
            else:
                print(f"RANK {local_rank}: nn.BatchNorm3d -> nn.SyncBatchNorm ({name}): train mode is {module.training}.")

    # Make sure all the Tensors in optimizer to be in the same device as model.
    if restore_total_step is not None:
        logging.info(f"Converting device state of Tensors in optimizer in GPUs [{local_rank}]...")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logging.info(f"Done converting device state of Tensors in optimizer in GPUs [{local_rank}].")
    
    # Ensure the synchronization.
    if restore_total_step is not None:
        dist.barrier()
    
    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision) if args.precision_dtype != "bfloat16" else None

    should_keep_training = True
    global_batch_num = 0 if restore_total_step is None else restore_total_step
    total_steps = 0 if restore_total_step is None else restore_total_step
    epoch = 0

    while should_keep_training:
        # Shuffle the data by using the sampler for each epoch.
        train_sampler.set_epoch(epoch)
        # Start the training.
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader, desc=f"Training: ", dynamic_ncols=True, disable=not is_main_process)):
            optimizer.zero_grad()
            left_img, right_img, disp_gt, valid = [x.to(device) for x in data_blob]

            assert model.training
            disp_init_pred = None
            disp_preds = model(left_img, right_img, iters=args.train_iters)
            if len(disp_preds) == 2:
                disp_init_pred = disp_preds[0]
                disp_preds = disp_preds[-1]
            else:
                disp_preds = disp_preds
            assert model.training

            loss = 0.0
            if disp_init_pred is not None:
                loss_init = 1.0 * smooth_l1_loss(disp_init_pred, disp_gt, valid, max_disp=args.max_disp)
                loss += loss_init
                if is_main_process and logger is not None:
                    logger.writer.add_scalar("loss/init_loss", loss_init.item(), global_batch_num)
                if i_batch == 0:
                    print(f"Add init loss in GPUs [{local_rank}].")
            loss_preds = sequence_loss(disp_preds, disp_gt, valid, max_disp=args.max_disp)
            loss += loss_preds
            if i_batch == 0:
                print(f"Add sequence loss in GPUs [{local_rank}].")
            metrics = get_metrics(disp_preds[-1], disp_gt, valid, max_disp=args.max_disp)

            if is_main_process and logger is not None:
                logger.writer.add_scalar("loss/preds_loss", loss_preds.item(), global_batch_num)
                logger.writer.add_scalar("loss/total_loss", loss.item(), global_batch_num)
                for i_param, param in enumerate(optimizer.param_groups):
                    logger.writer.add_scalar(f"lr/learning_rate_{i_param}", param["lr"], global_batch_num)
                logger.push(metrics)
            
            global_batch_num += 1

            if args.precision_dtype == "bfloat16":
                if i_batch == 0:
                    print(f"Not use GradScaler (value: {scaler}) when mixed precision with {args.precision_dtype} in GPUs [{local_rank}].")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            else:
                if i_batch == 0:
                    print(f"Use GradScaler (value: {scaler}) when mixed precision with {args.precision_dtype} in GPUs [{local_rank}].")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

            if total_steps % validation_frequency == validation_frequency - 1:
                if is_main_process:
                    save_path = Path(args.logdir + f"/{total_steps + 1}_{args.name}.pth")
                    optimizer_path = Path(args.logdir + f"/optimizer/{total_steps + 1}_{args.name}.pth")
                    scheduler_path = Path(args.logdir + f"/scheduler/{total_steps + 1}_{args.name}.pth")
                    logging.info(f"Saving file {save_path.absolute()}.")
                    torch.save(model.state_dict(), save_path)
                    torch.save(optimizer.state_dict(), optimizer_path)
                    torch.save(scheduler.state_dict(), scheduler_path)
                model.eval()
                epe_list, out_list, occ_epe_list, occ_out_list, non_occ_epe_list, non_occ_out_list = [], [], [], [], [], []
                for data in tqdm(val_loader, desc=f"Evaluating: ", dynamic_ncols=True, disable=not is_main_process):
                    (imageL_file, imageR_file, GT_file), left_img, right_img, disp_gt, valid = [x for x in data]
                    left_img = left_img.to(device)
                    right_img = right_img.to(device)
                    disp_gt = disp_gt.to(device)
                    valid = valid.to(device)

                    padder = InputPadder(left_img.shape, divis_by=32)
                    left_img, right_img = padder.pad(left_img, right_img)
                    with torch.no_grad():
                        outputs = model(left_img, right_img, iters=args.eval_iters, test_mode=True)
                    disp_pred = padder.unpad(outputs[1])
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    epe = torch.sum((disp_pred - disp_gt) ** 2, dim=1).sqrt().unsqueeze(1)
                    nan_mask = ~torch.isnan(epe)
                    out = (epe > 3.0).float()
                    
                    if val_dataset_name == "sceneflow" or "kitti" in val_dataset_name:
                        valid_mask = (valid.unsqueeze(1) >= 0.5) & (disp_gt.abs() < args.max_disp)
                        # Compute the occlusion and non-occlusion mask.
                        left_img = padder.unpad(left_img)
                        right_img = padder.unpad(right_img)
                        left_img = 2 * (left_img / 255.0) - 1.0
                        right_img = 2 * (right_img / 255.0) - 1.0
                        warped_right_img, mask = disp_warp(right_img, disp_gt)
                        error = (torch.abs(left_img - warped_right_img).mean(1, keepdim=True) < 0.03).float()
                        mask = error * mask
                        occ_mask = (1 - mask).bool() * valid_mask
                        nonocc_mask = mask.bool() * valid_mask
                        
                        occ_epe = epe[occ_mask & nan_mask].mean().item()
                        occ_out = out[occ_mask & nan_mask].mean().item()
                        non_occ_epe = epe[nonocc_mask & nan_mask].mean().item()
                        non_occ_out = out[nonocc_mask & nan_mask].mean().item()
                        epe = epe[valid_mask & nan_mask].mean().item()
                        out = out[valid_mask & nan_mask].mean().item()

                        epe_list.append(epe)
                        out_list.append(out)
                        occ_epe_list.append(occ_epe)
                        occ_out_list.append(occ_out)
                        non_occ_epe_list.append(non_occ_epe)
                        non_occ_out_list.append(non_occ_out)
                    elif "middlebury" in val_dataset_name or "eth3d" in val_dataset_name:
                        occ_mask = Image.open(GT_file[0].replace("disp0GT.pfm", "mask0nocc.png")).convert("L")
                        occ_mask = torch.from_numpy(np.ascontiguousarray(occ_mask)).to(epe.dtype).to(device)
                        valid_mask = (valid >= 0.5) & (occ_mask==255)

                        epe = epe[valid_mask & nan_mask].mean().item()
                        out = out[valid_mask & nan_mask].mean().item()

                        epe_list.append(epe)
                        out_list.append(out)
                        occ_epe_list.append(0)
                        occ_out_list.append(0)
                        non_occ_epe_list.append(0)
                        non_occ_out_list.append(0)
                    else:
                        raise NotImplementedError("Only sceneflow and kitti dataset can be evaluate during training.")
                
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
                        f"{val_dataset_name}-d3": total_out,
                        f"{val_dataset_name}-occ-epe": total_occ_epe,
                        f"{val_dataset_name}-occ-d3": total_occ_out,
                        f"{val_dataset_name}-nonocc-epe": total_non_occ_epe,
                        f"{val_dataset_name}-nonocc-d3": total_non_occ_out,
                    }

                    print(f"Evaluation {val_dataset_name}: EPE {round(total_epe, 4)}, D3 {round(total_out, 4)}, Occ-EPE {round(total_occ_epe, 4)}, Occ-D3 {round(total_occ_out, 4)}, Non-Occ-EPE {round(total_non_occ_epe, 4)}, Non-Occ-D3 {round(total_non_occ_out, 4)}.")

                    with open(args.logdir + f"/test_{dataset_name}.txt", "a") as file:
                        line = []
                        for key, value in results.items():
                            line.append(key + f": {round(value, 4)}")
                        file.write(" | ".join(line) + "\n")
                    if logger is not None:
                        logger.write_dict(results)
                model.train()
                model.module.freeze_bn()
                # Check whether the freezing pipeline is correct.
                for name, module in model.module.named_modules():
                    if isinstance(module, nn.SyncBatchNorm):
                        if name in model.module.freezing_module_list:
                            print(f"RANK [{local_rank}]: nn.BatchNorm2d -> nn.SyncBatchNorm ({name}): train mode is {module.training}.")
                        else:
                            print(f"RANK [{local_rank}]: nn.BatchNorm3d -> nn.SyncBatchNorm ({name}): train mode is {module.training}.")
                dist.barrier()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        
        epoch += 1
        
        if len(train_loader) >= 10000 and is_main_process:
            save_path = Path(args.logdir + f"/{total_steps + 1}_epoch_{args.name}.pth.gz")
            logging.info(f"Saving file {save_path}.")
            torch.save(model.state_dict(), save_path)
        
    print(f"FINISHED TRAINING in GPUs [{local_rank}].")
    if is_main_process:
        if logger is not None:
            logger.close()
        PATH = args.logdir + f"/{args.name}.pth"
        torch.save(model.state_dict(), PATH)
    
    # Destroy the processes.
    dist.destroy_process_group()

    return PATH if is_main_process else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="raft-stereo", help="name your experiment.")
    parser.add_argument("--restore_ckpt", default=None, help="load the weights from a specific checkpoint.")
    parser.add_argument("--mixed_precision", default=True, action="store_true", help="use mixed precision.")
    parser.add_argument("--precision_dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="choose mixed precision type: float16, bfloat16 or float32.")
    parser.add_argument("--logdir", default="./checkpoints/sceneflow", help="the directory to save logs and checkpoints.")
    parser.add_argument("--backbone_type", required=False, help="the type of the backbone used in the network ('MobileNetV2' or 'ResidualNet' for default if not set this argument explicitly).")
    parser.add_argument("--backbone_ckpt", required=False, help="the path of the backbone checkpoint.")

    # Training parameters.
    parser.add_argument("--optimizer", type=str, default="adamw", help="name of the optimizer used during training.")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size used during training.")
    parser.add_argument("--train_datasets", nargs="+", default=["sceneflow"], help="training datasets.")
    parser.add_argument("--train_datasets_root", nargs="+", default=["/data/sceneflow/"], help="training datasets roots.")
    parser.add_argument("--lr", type=float, default=0.0002, help="max learning rate.")
    parser.add_argument("--num_steps", type=int, default=200000, help="length of training schedule.")
    parser.add_argument("--image_size", type=int, nargs="+", default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument("--train_iters", type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument("--wdecay", type=float, default=0.00001, help="weight decay in optimizer.")
    parser.add_argument("--seed", type=int, default=666, help="random seed for the whole system.")

    # Evaluation parameters.
    parser.add_argument("--eval_iters", type=int, default=32, help="number of disparity field updates during evaluation forward pass.")

    # Architecture choices.
    parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders.")
    parser.add_argument("--cv_levels", type=int, default=4, help="number of levels in the cost volume pyramid.")
    parser.add_argument("--cv_radius", type=int, default=4, help="width of the cost volume pyramid.")
    parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1 / 2 ^ k).")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently.")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels.")
    parser.add_argument("--channels", nargs="+", type=int, default=[128] * 3, help="hidden state and context channels.")
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder.")
    parser.add_argument("--max_disp", type=int, default=192, help="max disparity.")

    # Data augmentation.
    parser.add_argument("--img_gamma", type=float, nargs="+", default=None, help="gamma range.")
    parser.add_argument("--saturation_range", type=float, nargs="+", default=[0.0, 1.4], help="color saturation.")
    parser.add_argument("--do_flip", default=False, choices=["h", "v"], help="flip the images horizontally or vertically.")
    parser.add_argument("--spatial_scale", type=float, nargs="+", default=[-0.2, 0.4], help="re-scale the images randomly.")
    parser.add_argument("--noyjitter", action="store_true", help="do not simulate imperfect rectification.")

    args = parser.parse_args()
    
    # Create the log directories.
    if torch.multiprocessing.current_process().name == "MainProcess":
        Path(args.logdir).mkdir(exist_ok=True, parents=True)
        Path(args.logdir + "/optimizer").mkdir(exist_ok=True, parents=True)
        Path(args.logdir + "/scheduler").mkdir(exist_ok=True, parents=True)
    
    # Obtain the world size (number of GPUs).
    world_size = torch.cuda.device_count()
    if torch.multiprocessing.current_process().name == "MainProcess":
        print(f"Using {world_size} GPUs for Distributed Data Parallel training.")
    
    # Start the Distributed Data Parallel training.
    torch.multiprocessing.spawn(
        train_dist,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
