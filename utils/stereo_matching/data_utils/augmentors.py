import os
import cv2
import glob
import random
import warnings
import numpy as np
import torch
from torchvision.transforms import ColorJitter, Compose, functional
from PIL import Image
from skimage import color
from typing import List, Tuple, Union

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_kitti_images() -> list:
    return sorted(glob.glob("datasets/KITTI/training/image_2/*_10.png"))


def get_eth3d_images() -> list:
    return sorted(glob.glob("datasets/ETH3D/two_view_training/*/im0.png"))


def get_middlebury_images() -> list:
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), "r") as file:
        lines = file.read().splitlines()
    return sorted([os.path.join(root, "trainingQ", f"{name}/im0.png") for name in lines])


def transfer_color(image: np.ndarray, style_mean: float, style_stddev: float) -> np.ndarray:
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0, 1), keepdims=True)
    reference_mean = np.mean(reference_image_lab, axis=(0, 1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev / reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l, a, b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb


class AdjustGamma(object):
    def __init__(self, gamma_min: Union[int, float], gamma_max: Union[int, float], gain_min: Union[int, float]=1.0, gain_max: Union[int, float]=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max
    
    def __call__(self, sample: Image.Image) -> Union[torch.Tensor, Image.Image]:
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)
    
    def __repr__(self) -> str:
        return f"Adjust Gamma ({self.gamma_min}, {self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})."


class DispAugmentor:
    def __init__(
        self,
        crop_size: Union[List[Union[int, float]], Tuple[Union[int, float]]],
        min_scale: float=-0.2,
        max_scale: float=0.5,
        do_flip: bool=True,
        yjitter: bool=False,
        saturation_range: List[float]=[0.6, 1.4],
        gamma: List[Union[int, float]]=[1, 1, 1, 1],
    ):
        # Spatial augmentation params.
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params.
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params.
        self.photo_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
    
    def color_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        """ Photometric augmentation. """

        # Asymmetric.
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # Symmetric.
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        
        return img1, img2
    
    def eraser_transform(self, img1: np.ndarray, img2: np.ndarray, bounds: List[int]=[50, 100]) -> Tuple[np.ndarray]:
        """ Occlusion augmentation. """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        
        return img1, img2
    
    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray) -> Tuple[np.ndarray]:
        # Randomly sample scale.

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd),
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # Rescale the images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            disp = disp * [scale_x, scale_y]
        
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == "hf": # h-flip.
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                disp = disp[:, ::-1] * [-1.0, 1.0]
            
            if np.random.rand() < self.h_flip_prob and self.do_flip == "h": # h-flip for stereo.
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp
            
            if np.random.rand() < self.v_flip_prob and self.do_flip == "v": # v-flip.
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                disp = disp[::-1, :] * [1.0, -1.0]
        
        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        
        return img1, img2, disp
    
    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray) -> Tuple[np.ndarray]:
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, disp = self.spatial_transform(img1, img2, disp)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp)

        return img1, img2, disp


class SparseDispAugmentor:
    def __init__(
        self,
        crop_size: Union[List[Union[int, float]], Tuple[Union[int, float]]],
        min_scale: float=-0.2,
        max_scale: float=0.5,
        do_flip: bool=False,
        yjitter: bool=False,
        saturation_range: List[float]=[0.7, 1.3],
        gamma: List[Union[int, float]]=[1, 1, 1, 1],
    ):
        # Spatial augmentation params.
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params.
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params.
        self.photo_aug = Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
    
    def color_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2
    
    def eraser_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        
        return img1, img2
    
    def resize_sparse_disp_map(self, disp: np.ndarray, valid: np.ndarray, fx: float=1.0, fy: float=1.0) -> Tuple[np.ndarray]:
        ht, wd = disp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)
        disp = disp.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        disp0 = disp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp1 = disp0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp1 = disp1[v]

        disp_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_img[yy, xx] = disp1
        valid_img[yy, xx] = 1

        return disp_img, valid_img
    
    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray]:
        # Randomly sample scale.
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd),
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # Rescale the images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp, valid = self.resize_sparse_disp_map(disp, valid, fx=scale_x, fy=scale_y)
        
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == "hf": # h-flip.
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                disp = disp[:, ::-1] * [-1.0, 1.0]
            
            if np.random.rand() < self.h_flip_prob and self.do_flip == "h": # h-flip for stereo.
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp
            
            if np.random.rand() < self.v_flip_prob and self.do_flip == "v": # v-flip.
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                disp = disp[::-1, :] * [1.0, -1.0]
        
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, disp, valid

    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray]:
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, disp, valid = self.spatial_transform(img1, img2, disp, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp)
        valid = np.ascontiguousarray(valid)

        return img1, img2, disp, valid
