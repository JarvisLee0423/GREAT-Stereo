import torch
import os
import cv2
import glob
import time
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from utils.utils import InputPadder
from utils.stereo_matching.data_utils import readers


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def demo(args):
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
    model = torch.nn.DataParallel(module(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(args.output_directory.replace("eval", "vis"), exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {args.output_directory}/")

        for i, (imfile1, imfile2) in enumerate(tqdm(list(zip(left_images, right_images)))):
            image1 = torch.from_numpy(np.array(readers.readGen(imfile1)).astype(np.uint8)).permute(2, 0, 1).float()[None].to(DEVICE)
            image2 = torch.from_numpy(np.array(readers.readGen(imfile2)).astype(np.uint8)).permute(2, 0, 1).float()[None].to(DEVICE)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            # Warmup to measure the runtime.
            if i == 0:
                for _ in range(5):
                    model(image1, image2, iters=args.valid_iters, test_mode=True)

            time_start = time.perf_counter()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            runtime = time.perf_counter() - time_start
            disp = padder.unpad(disp[1])
            file_stem = os.path.join(args.output_directory, f"{imfile1.split('/')[-2]}.png")
            disp = disp.cpu().numpy().squeeze()

            if args.save_pfm:
                image1_vis = cv2.cvtColor(np.uint8(padder.unpad(image1).squeeze().permute(1, 2, 0).detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                disp_192_vis = cv2.applyColorMap(np.uint8(disp / 192.0 * 255.0), cv2.COLORMAP_JET)
                disp_max_vis = cv2.applyColorMap(np.uint8(disp / disp.max() * 255.0), cv2.COLORMAP_JET)
                vis = np.concatenate([image1_vis, disp_192_vis, disp_max_vis], axis=0)
                cv2.imwrite(file_stem.replace("eval", "vis"), vis)

                readers.writePFM(file_stem.replace(".png", ".pfm"), disp)
                with open(file_stem.replace(".png", ".txt"), "w") as file:
                    file.write("runtime " + str(runtime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="great-igev-stereo", help="name your experiment.")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/eth3d/eth3d_finetune.pth')
    parser.add_argument("--backbone_type", required=False, help="the type of the backbone used in the network ('MobileNetV2' or 'ResidualNet' for default if not set this argument explicitly).")
    parser.add_argument("--backbone_ckpt", required=False, help="the path of the backbone checkpoint.")
    parser.add_argument('--save_pfm', action='store_true', default=True, help='save output as pfm file')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="~/Workspaces/Researches/Datasets/ETH3D/two_view_*/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="~/Workspaces/Researches/Datasets/ETH3D/two_view_*/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./experiments/great_stereo/igev-based/eth3d/finetune/great+full+eth3d+finetune/eval/low_res_two_view")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--precision_dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="choose mixed precision type: float16, bfloat16 or float32.")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders.")
    parser.add_argument('--cv_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--cv_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently.")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--channels", nargs="+", type=int, default=[128] * 3, help="hidden state and context channels.")
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)
