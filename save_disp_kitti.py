import torch
import os
import cv2
import glob
import argparse
import importlib
import skimage.io
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils import InputPadder


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


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

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp[1])
            file_stem = os.path.join(args.output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            # plt.imsave(file_stem, disp, cmap='jet')
            if args.save_png:
                image1_vis = cv2.cvtColor(np.uint8(padder.unpad(image1).squeeze().permute(1, 2, 0).detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                disp_192_vis = cv2.applyColorMap(np.uint8(disp / 192.0 * 255.0), cv2.COLORMAP_JET)
                disp_max_vis = cv2.applyColorMap(np.uint8(disp / disp.max() * 255.0), cv2.COLORMAP_JET)
                vis = np.concatenate([image1_vis, disp_192_vis, disp_max_vis], axis=0)
                
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(file_stem, disp_16)
                cv2.imwrite(file_stem.replace("eval", "vis"), vis)

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="great-igev-stereo", help="name your experiment.")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti/kitti15.pth')
    parser.add_argument("--backbone_type", required=False, help="the type of the backbone used in the network ('MobileNetV2' or 'ResidualNet' for default if not set this argument explicitly).")
    parser.add_argument("--backbone_ckpt", required=False, help="the path of the backbone checkpoint.")
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="~/Workspaces/Researches/Datasets/KITTI/2015/data_scene_flow/testing/image_2/*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="~/Workspaces/Researches/Datasets/KITTI/2015/data_scene_flow/testing/image_3/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="~/Workspaces/Researches/Datasets/KITTI/2012/data_stereo_flow/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="~/Workspaces/Researches/Datasets/KITTI/2012/data_stereo_flow/testing/colored_1/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./experiments/great_stereo/igev-based/kitti/finetune/2015/great+full+kitti/eval/disp_0")
    # parser.add_argument('--output_directory', help="directory to save output", default="./experiments/great_stereo/igev-based/kitti/finetune/2012/great+full+kitti/eval/disp_0")
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
