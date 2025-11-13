import argparse
import logging
import torch.utils.data as data
from typing import Union
from utils.stereo_matching.data_utils.datasets import *


def fetch_training_data(args: argparse.Namespace, dataset_only=False) -> Union[data.Dataset, data.DataLoader]:
    """ Create the data loader for the corresponding training set. """

    aug_params = {"crop_size": args.image_size, "min_scale": args.spatial_scale[0], "max_scale": args.spatial_scale[1], "do_flip": False, "yjitter": not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
    
    train_dataset = None
    for dataset_name, dataset_root in zip(args.train_datasets, args.train_datasets_root):
        if dataset_name == "sceneflow":
            # clean_dataset = SceneFlowStereoDataset(aug_params, dataset_root, dstype="frames_cleanpass")
            # final_dataset = SceneFlowStereoDataset(aug_params, dataset_root, dstype="frames_finalpass")
            # new_dataset = (clean_dataset * 4) + (final_dataset * 4)
            new_dataset = SceneFlowStereoDataset(aug_params, dataset_root, dstype="frames_finalpass")
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow.")
        elif dataset_name == "vkitti2":
            new_dataset = VisualKITTI2StereoDataset(aug_params, dataset_root)
            logging.info(f"Adding {len(new_dataset)} samples from Visual KITTI 2.")
        elif dataset_name == "kitti":
            kitti12 = KITTIStereoDataset(aug_params, f"{dataset_root}/2012/data_stereo_flow")
            kitti15 = KITTIStereoDataset(aug_params, f"{dataset_root}/2015/data_scene_flow")
            new_dataset = kitti12 + kitti15
            logging.info(f"Adding {len(kitti12)} samples from KITTI 2012.")
            logging.info(f"Adding {len(kitti15)} samples from KITTI 2015.")
            logging.info(f"Adding {len(new_dataset)} samples from KITTI.")
        elif dataset_name == "booster":
            crestereo = CREStereoDataset(aug_params, f"{dataset_root}/CREStereo")
            instereo2k = InStereo2KStereoDataset(aug_params, f"{dataset_root}/InStereo2K")
            hrvs = HRVSStereoDataset(aug_params, f"{dataset_root}/HRVS")
            balanced = BoosterStereoDataset(aug_params, f"{dataset_root}/Booster", split="balanced")
            fallingthings = FallingThingsStereoDataset(aug_params, f"{dataset_root}/FallingThings")
            new_dataset = crestereo + instereo2k * 50 + hrvs * 50 + balanced * 200 + fallingthings * 5
            logging.info(f"Adding {len(crestereo)} samples from CREStereo.")
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K.")
            logging.info(f"Adding {len(hrvs)} samples from HRVS.")
            logging.info(f"Adding {len(balanced)} samples from Booster Balanced.")
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings.")
            logging.info(f"Adding {len(new_dataset)} samples from Booster Mixture Dataset.")
        elif dataset_name == "eth3d_train":
            tartanair = TartanAirStereoDataset(aug_params, f"{dataset_root}/TartanAir")
            sceneflow = SceneFlowStereoDataset(aug_params, f"{dataset_root}/SceneFlow", dstype="frames_finalpass")
            sintel = SintelStereoDataset(aug_params, f"{dataset_root}/Sintel")
            crestereo = CREStereoDataset(aug_params, f"{dataset_root}/CREStereo")
            eth3d = ETH3DStereoDataset(aug_params, f"{dataset_root}/ETH3D")
            instereo2k = InStereo2KStereoDataset(aug_params, f"{dataset_root}/InStereo2K")
            new_dataset = tartanair + sceneflow + sintel * 50 + eth3d * 1000 + instereo2k * 100 + crestereo * 2
            logging.info(f"Adding {len(tartanair)} samples from Tartan Air.")
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow.")
            logging.info(f"Adding {len(sintel)} samples from Sintel.")
            logging.info(f"Adding {len(crestereo)} samples from CREStereo.")
            logging.info(f"Adding {len(eth3d)} samples from ETH3D.")
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K.")
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset.")
        elif dataset_name == "eth3d_finetune":
            crestereo = CREStereoDataset(aug_params, f"{dataset_root}/CREStereo")
            eth3d = ETH3DStereoDataset(aug_params, f"{dataset_root}/ETH3D")
            instereo2k = InStereo2KStereoDataset(aug_params, f"{dataset_root}/InStereo2K")
            new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
            logging.info(f"Adding {len(crestereo)} samples from CREStereo.")
            logging.info(f"Adding {len(eth3d)} samples from ETH3D.")
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K.")
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset.")
        elif dataset_name == "middlebury_train":
            tartanair = TartanAirStereoDataset(aug_params, f"{dataset_root}/TartanAir")
            sceneflow = SceneFlowStereoDataset(aug_params, f"{dataset_root}/SceneFlow", dstype="frames_finalpass")
            fallingthings = FallingThingsStereoDataset(aug_params, f"{dataset_root}/FallingThings")
            hrvs = HRVSStereoDataset(aug_params, f"{dataset_root}/HRVS")
            crestereo = CREStereoDataset(aug_params, f"{dataset_root}/CREStereo")
            instereo2k = InStereo2KStereoDataset(aug_params, f"{dataset_root}/InStereo2K")
            mb2005 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2005")
            mb2006 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2006")
            mb2014 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2014")
            mb2021 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2021")
            mbeval3 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="MiddEval3", resolution="H")
            new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + hrvs * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
            logging.info(f"Adding {len(tartanair)} samples from Tartan Air.")
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow.")
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings.")
            logging.info(f"Adding {len(hrvs)} samples from HRVS.")
            logging.info(f"Adding {len(crestereo)} samples from CREStereo.")
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K.")
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005.")
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006.")
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014.")
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021.")
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3.")
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset.")
        elif dataset_name == "middlebury_finetune":
            crestereo = CREStereoDataset(aug_params, f"{dataset_root}/CREStereo")
            instereo2k = InStereo2KStereoDataset(aug_params, f"{dataset_root}/InStereo2K")
            hrvs = HRVSStereoDataset(aug_params, f"{dataset_root}/HRVS")
            mb2005 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2005")
            mb2006 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2006")
            mb2014 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2014")
            mb2021 = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="2021")
            mbeval3_h = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="MiddEval3", resolution="H")
            mbeval3_f = MiddleburyStereoDataset(aug_params, f"{dataset_root}/Middlebury", split="MiddEval3", resolution="F")
            fallingthings = FallingThingsStereoDataset(aug_params, f"{dataset_root}/FallingThings")
            new_dataset = crestereo + instereo2k * 50 + hrvs * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3_h * 200 + mbeval3_f * 200 + fallingthings * 5
            logging.info(f"Adding {len(crestereo)} samples from CREStereo.")
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K.")
            logging.info(f"Adding {len(hrvs)} samples from HRVS.")
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005.")
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006.")
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014.")
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021.")
            logging.info(f"Adding {len(mbeval3_h)} samples from Middlebury Eval3 Half.")
            logging.info(f"Adding {len(mbeval3_f)} samples from Middlebury Eval3 Full.")
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings.")
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset.")
        else:
            new_dataset = None
            logging.info(f"Dataset named {dataset_name} has no defined datasets!")
        
        if new_dataset is not None:
            if train_dataset is None:
                train_dataset = new_dataset
            else:
                train_dataset = train_dataset + new_dataset
    
    logging.info(f"Training with {len(train_dataset)} image pairs.")

    if dataset_only:
        print("Only return the dataset.")
        
        return train_dataset
    else:
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 2,
            drop_last=True,
        )

        return train_loader
