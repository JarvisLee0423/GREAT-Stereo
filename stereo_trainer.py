from __future__ import division, print_function
import logging
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from utils.stereo_matching.test_utils.evaluators import *
from utils.stereo_matching.train_utils.loggers import Logger
from utils.stereo_matching.train_utils.losses import get_metrics, smooth_l1_loss, sequence_loss
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


def train(args: argparse.Namespace) -> str:
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
    model = nn.DataParallel(module(args))
    print(f"All Parameter Count: {count_all_parameters(model)}.")
    print(f"Grad Required Parameter Count: {count_grad_required_parameters(model)}.")

    # Save the training settings.
    args_dict = vars(args)
    with open(args.logdir + "/training_settings.txt", "w") as file:
        for key, value in args_dict.items():
            file.write("{:<20} {:<10}".format(key, str(value)) + "\n")
    
    # Save the model settings.
    with open(args.logdir + "/model_settings.txt", "w") as file:
        file.write("All Model Parameters: " + str(count_all_parameters(model)) + "\n")
        file.write("Grad Required Model Parameters: " + str(count_grad_required_parameters(model)) + "\n")
        file.write(model.module.__str__())

    # Get training data.
    train_loader = fetch_training_data(args)

    # Get optimizer and scheduler.
    optim_name = args.optimizer
    fetch_optimizer_func = getattr(importlib.import_module("utils.stereo_matching.train_utils.optimizers"), f"fetch_{optim_name}_optimizer")
    optimizer, scheduler = fetch_optimizer_func(args, model)

    # Get logger.
    logger = Logger(args, model, scheduler)

    # Load the checkpoint, if provide.
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint.")
    
    # Start training the model.
    model.cuda()
    model.train()
    # We keep BatchNorm frozen.
    model.module.freeze_bn()

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    total_steps = 0

    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            left_img, right_img, disp_gt, valid = [x.cuda() for x in data_blob]

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
                logger.writer.add_scalar("init_loss", loss_init.item(), global_batch_num)
                if i_batch == 0:
                    print("Add init loss.")
            loss_preds = sequence_loss(disp_preds, disp_gt, valid, max_disp=args.max_disp)
            loss += loss_preds
            if i_batch == 0:
                print("Add sequence loss.")
            metrics = get_metrics(disp_preds[-1], disp_gt, valid, max_disp=args.max_disp)

            logger.writer.add_scalar("preds_loss", loss_preds.item(), global_batch_num)
            logger.writer.add_scalar("total_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_batch_num)
            logger.push(metrics)

            global_batch_num += 1
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(args.logdir + f"/{total_steps + 1}_{args.name}.pth")
                logging.info(f"Saving file {save_path.absolute()}.")
                torch.save(model.state_dict(), save_path)
                for dataset_name, dataset_root in zip(args.train_datasets, args.train_datasets_root):
                    if dataset_name == "sceneflow":
                        results = evaluate_sceneflow(model.module, dataset_root, iters=args.eval_iters)
                    elif "kitti" in dataset_name:
                        results = evaluate_kitti(model.module, "/data/KITTI/2015/data_scene_flow", iters=args.eval_iters)
                    elif "middlebury" in dataset_name:
                        results = evaluate_middlebury(model.module, "/data/Middlebury", iters=args.eval_iters, split="H")
                    elif "eth3d" in dataset_name:
                        results = evaluate_eth3d(model.module, "/data/ETH3D", iters=args.eval_iters)
                    else:
                        print("Only sceneflow and kitti dataset can be evaluate during training.")
                        results = None
                        continue

                    with open(args.logdir + f"/test_{dataset_name}.txt", "a") as file:
                        line = []
                        for key, value in results.items():
                            line.append(key + f": {round(value, 4)}")
                        file.write(" | ".join(line) + "\n")
                
                if results is not None:
                    logger.write_dict(results)
                model.train()
                model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        
        if len(train_loader) >= 10000:
            save_path = Path(args.logdir + f"/{total_steps + 1}_epoch_{args.name}.pth.gz")
            logging.info(f"Saving file {save_path}.")
            torch.save(model.state_dict(), save_path)
    
    print("FINISHED TRAINING.")
    logger.close()
    PATH = args.logdir + f"/{args.name}.pth"
    torch.save(model.state_dict(), PATH)

    return PATH


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
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
    parser.add_argument("--max_disp", type=int, default=192, help="max disparity.")

    # Data augmentation.
    parser.add_argument("--img_gamma", type=float, nargs="+", default=None, help="gamma range.")
    parser.add_argument("--saturation_range", type=float, nargs="+", default=[0.0, 1.4], help="color saturation.")
    parser.add_argument("--do_flip", default=False, choices=["h", "v"], help="flip the images horizontally or vertically.")
    parser.add_argument("--spatial_scale", type=float, nargs="+", default=[-0.2, 0.4], help="re-scale the images randomly.")
    parser.add_argument("--noyjitter", action="store_true", help="do not simulate imperfect rectification.")
    
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
