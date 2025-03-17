import os
import copy
import glob
import random
import logging
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Union
from utils.stereo_matching.data_utils import readers
from utils.stereo_matching.data_utils.augmentors import DispAugmentor, SparseDispAugmentor


class BaseStereoDataset(data.Dataset):
    def __init__(self, aug_params: dict=None, sparse: bool=False, reader: Any=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseDispAugmentor(**aug_params)
            else:
                self.augmentor = DispAugmentor(**aug_params)
        
        if reader is None:
            self.disparity_reader = readers.readGen
        else:
            self.disparity_reader = reader
        
        self.is_test = False
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
    
    def __getitem__(self, index: int) -> Union[list, torch.Tensor]:
        if self.is_test:
            img1 = readers.readGen(self.image_list[index][0])
            img2 = readers.readGen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disp_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024
        
        img1 = readers.readGen(self.image_list[index][0])
        img2 = readers.readGen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # Grayscale images.
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)
        
        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)
        
        disp = flow[:1]

        return self.image_list[index] + [self.disp_list[index]], img1, img2, disp, valid.float()
    
    def __mul__(self, v: Union[int, float]) -> Any:
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disp_list = v * copy_of_self.disp_list
        copy_of_self.extra_info = v * copy_of_self.extra_info

        return copy_of_self
    
    def __len__(self) -> int:
        return len(self.image_list)


class ETH3DStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/ETH3D/", split: str="training"):
        super(ETH3DStereoDataset, self).__init__(aug_params, sparse=True)

        left_images = sorted(glob.glob(os.path.join(root, f"two_view_{split}/*/im0.png")))
        right_images = sorted(glob.glob(os.path.join(root, f"two_view_{split}/*/im1.png")))
        disp_images = sorted(glob.glob(os.path.join(root, "two_view_training_gt/*/disp0GT.pfm"))) if split == "training" else [os.path.join(root, "two_view_training_gt/playground_1l/disp0GT.pfm")] * len(left_images)

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class SintelStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/SintelStereo/"):
        super(SintelStereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispSintelStereo)

        left_images = sorted(glob.glob(os.path.join(root, "training/*_left/*/frame_*.png")))
        right_images = sorted(glob.glob(os.path.join(root, "training/*_right/*/frame_*.png")))
        disp_images = sorted(glob.glob(os.path.join(root, "training/disparities/*/frame_*.png"))) * 2

        for left, right, disp in zip(left_images, right_images, disp_images):
            assert left.split("/")[-2:] == disp.split("/")[-2:]
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class FallingThingsStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/FallingThings/"):
        super(FallingThingsStereoDataset, self).__init__(aug_params, reader=readers.readDispFallingThings)
        
        assert os.path.exists(root)

        left_images = sorted(glob.glob(root + "/mixed/*/*left.jpg") + glob.glob(root + "/single/*/*/*left.jpg"))
        right_images = sorted(glob.glob(root + "/mixed/*/*right.jpg") + glob.glob(root + "/single/*/*/*right.jpg"))
        disp_images = sorted(glob.glob(root + "/mixed/*/*left.depth.png") + glob.glob(root + "/single/*/*/*left.depth.png"))

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class TartanAirStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/TartanAir/"):
        super(TartanAirStereoDataset, self).__init__(aug_params, reader=readers.readDispTartanAir)

        assert os.path.exists(root)

        left_images = sorted(glob.glob(root + "/cameras/left/*/*/*/*/image_left/*.png"))
        right_images = sorted(glob.glob(root + "/cameras/right/*/*/*/*/image_right/*.png"))
        disp_images = sorted(glob.glob(root + "/depth/*/*/*/*/depth_left/*.npy"))

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class CREStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/CREStereo/"):
        super(CREStereoDataset, self).__init__(aug_params, reader=readers.readDispCREStereo)

        assert os.path.exists(root)

        left_images = sorted(glob.glob(root + "/*/*/*_left.jpg"))
        right_images = sorted(glob.glob(root + "/*/*/*_right.jpg"))
        disp_images = sorted(glob.glob(root + "/*/*/*_left.disp.png"))

        for idx, (left, right, disp) in enumerate(zip(left_images, right_images, disp_images)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class HRVSStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/HRVS/"):
        super(HRVSStereoDataset, self).__init__(aug_params)

        assert os.path.exists(root)

        left_images = sorted(glob.glob(root + "/carla-highres/trainingF/*/im0.png"))
        right_images = sorted(glob.glob(root + "/carla-highres/trainingF/*/im1.png"))
        disp_images = sorted(glob.glob(root + "/carla-highres/trainingF/*/disp0GT.pfm"))

        for idx, (left, right, disp) in enumerate(zip(left_images, right_images, disp_images)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class InStereo2KStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/InStereo2K/"):
        super(InStereo2KStereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispInStereo2K)

        assert os.path.exists(root)

        left_images = sorted(glob.glob(root + "/train/*/*/left.png") + glob.glob(root + "/test/*/left.png"))
        right_images = sorted(glob.glob(root + "/train/*/*/right.png") + glob.glob(root + "/test/*/right.png"))
        disp_images = sorted(glob.glob(root + "/train/*/*/left_disp.png") + glob.glob(root + "/test/*/left_disp.png"))

        for idx, (left, right, disp) in enumerate(zip(left_images, right_images, disp_images)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class KITTIStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/KITTI/KITTI_2015/", image_set: str="training"):
        super(KITTIStereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispKITTI)

        assert os.path.exists(root)

        left_filename = "image_2" if "2015" in root else "colored_0"
        right_filename = "image_3" if "2015" in root else "colored_1"
        disp_filename = "disp_occ_0" if "2015" in root else "disp_occ"

        left_images = sorted(glob.glob(os.path.join(root, image_set, f"{left_filename}/*_10.png")))
        right_images = sorted(glob.glob(os.path.join(root, image_set, f"{right_filename}/*_10.png")))
        disp_images = sorted(glob.glob(os.path.join(root, "training", f"{disp_filename}/*_10.png"))) if image_set == "training" else [os.path.join(root, f"training/{disp_filename}/000085_10.png")] * len(left_images)

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class VisualKITTI2StereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/KITTI/Visual"):
        super(VisualKITTI2StereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispVKITTI2)

        assert os.path.exists(root)

        left_images = sorted(glob.glob(os.path.join(root, "2.0.3/vkitti_2.0.3_rgb/Scene*/*/frames/rgb/Camera_0/rgb*.jpg")))
        right_images = sorted(glob.glob(os.path.join(root, "2.0.3/vkitti_2.0.3_rgb/Scene*/*/frames/rgb/Camera_1/rgb*.jpg")))
        disp_images = sorted(glob.glob(os.path.join(root, "2.0.3/vkitti_2.0.3_depth/Scene*/*/frames/depth/Camera_0/depth*.png")))

        assert len(left_images) == len(right_images) == len(disp_images)

        for idx, (left, right, disp) in enumerate(zip(left_images, right_images, disp_images)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class BoosterStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/Booster", split: str="balanced"):
        super(BoosterStereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispBooster)

        assert os.path.exists(root)

        if split == "balanced":
            left_images = sorted(glob.glob(os.path.join(root, f"{split}/*/camera_00/im*.png")))
            right_images = sorted(glob.glob(os.path.join(root, f"{split}/*/camera_02/im*.png")))
            disp_images = [path.replace(f"camera_00/{os.path.basename(path)}", "disp_00.npy") for path in left_images]
        elif split == "unbalanced":
            left_images = sorted(glob.glob(os.path.join(root, f"{split}/*/camera_00/im*.png")))
            right_images = sorted(glob.glob(os.path.join(root, f"{split}/*/camera_01/im*.png")))
            disp_images = [path.replace(f"camera_00/{os.path.basename(path)}", "disp_00.npy") for path in left_images]
        else:
            raise ValueError("Wrong value for split. Only accept 'balanced' and 'unbalanced'.")
        
        assert len(left_images) == len(right_images) == len(disp_images) > 0, [len(left_images), len(right_images), len(disp_images)]

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]


class MiddleburyStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/Middlebury/", split: str="2014", resolution: str="F", test_set: bool=False):
        super(MiddleburyStereoDataset, self).__init__(aug_params, sparse=True, reader=readers.readDispMiddlebury)

        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disp_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disp_list += [str(scene / "disp1.png")]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disp_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disp_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*/*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disp_list += [str(scene / "disp0.pfm")]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disp_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [[str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disp_list += [str(scene / "disp0.pfm")]
        else:
            if test_set:
                left_images = sorted(glob.glob(root + f"/MiddEval3/test/test{resolution}/*/im0.png"))
                right_images = sorted(glob.glob(root + f"/MiddEval3/test/test{resolution}/*/im1.png"))
                disp_images = sorted(glob.glob(root + f"/MiddEval3/training/gt/gt0/training{resolution}/*/disp0GT.pfm"))
            else:
                left_images = sorted(glob.glob(root + f"/MiddEval3/training/training{resolution}/*/im0.png"))
                right_images = sorted(glob.glob(root + f"/MiddEval3/training/training{resolution}/*/im1.png"))
                disp_images = sorted(glob.glob(root + f"/MiddEval3/training/gt/gt0/training{resolution}/*/disp0GT.pfm"))

            assert len(left_images) == len(right_images) == len(disp_images) > 0, [left_images, split]

            for left, right, disp in zip(left_images, right_images, disp_images):
                self.image_list += [[left, right]]
                self.disp_list += [disp]


class SceneFlowStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params: dict=None, root: str="/data/SceneFlow/", dstype: str="frames_finalpass", things_test: bool=False):
        super(SceneFlowStereoDataset, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self.add_things("TEST")
        else:
            self.add_things("TRAIN")
            self.add_monkaa()
            self.add_driving()
    
    def add_things(self, split: str="TRAIN") -> None:
        """ Add FlyingThings3D data. """

        original_length = len(self.disp_list)
        root = os.path.join(self.root, "FlyingThings3D")
        left_images = sorted(glob.glob(os.path.join(root, self.dstype, split, "*/*/left/*.png")))
        right_images = [left.replace("left", "right") for left in left_images]
        disp_images = [left.replace(self.dstype, "disparity").replace(".png", ".pfm") for left in left_images]

        # Choose a random subset of 400 images for validation.
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images)))
        # val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (left, right, disp) in enumerate(zip(left_images, right_images, disp_images)):
            if (split == "TEST" and idx in val_idxs) or split == "TRAIN":
                self.image_list += [[left, right]]
                self.disp_list += [disp]
        
        logging.info(f"Added {len(self.disp_list) - original_length} from FlyingThings3D {self.dstype}.")
    
    def add_monkaa(self) -> None:
        """ Add Monkaa data. """

        original_length = len(self.disp_list)
        root = os.path.join(self.root, "Monkaa")
        left_images = sorted(glob.glob(os.path.join(root, self.dstype, "*/left/*.png")))
        right_images = [left.replace("left", "right") for left in left_images]
        disp_images = [left.replace(self.dstype, "disparity").replace(".png", ".pfm") for left in left_images]

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]
        
        logging.info(f"Added {len(self.disp_list) - original_length} from Monkaa {self.dstype}.")
    
    def add_driving(self) -> None:
        """ Add Driving data. """

        original_length = len(self.disp_list)
        root = os.path.join(self.root, "Driving")
        left_images = sorted(glob.glob(os.path.join(root, self.dstype, "*/*/*/left/*.png")))
        right_images = [left.replace("left", "right") for left in left_images]
        disp_images = [left.replace(self.dstype, "disparity").replace(".png", ".pfm") for left in left_images]

        for left, right, disp in zip(left_images, right_images, disp_images):
            self.image_list += [[left, right]]
            self.disp_list += [disp]
        
        logging.info(f"Added {len(self.disp_list) - original_length} from Driving {self.dstype}.")
