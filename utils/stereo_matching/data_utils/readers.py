import re
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readDispCREStereo(filename: str) -> np.ndarray:
    disp = np.array(Image.open(filename))
    return disp.astype(np.float32) / 32.0


def readDispBooster(filename: str) -> Tuple[np.ndarray]:
    disp = np.load(filename, encoding="bytes", allow_pickle=True)
    valid = disp > 0
    return disp, valid


def readDispKITTI(filename: str) -> Tuple[np.ndarray]:
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid


def readDispVKITTI2(filename: str) -> Tuple[np.ndarray]:
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = (depth / 100).astype(np.float32)
    valid = (depth > 0) & (depth < 655)
    focal_length = 725.0087
    baseline = 0.532725
    disp = baseline * focal_length / depth
    disp[~valid] = 0.0

    return disp, valid


def writeFlowKITTI(filename: str, uv: np.ndarray) -> None:
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def readFlowKITTI(filename: str) -> Tuple[np.ndarray]:
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def readDispInStereo2K(filename: str) -> Tuple[np.ndarray]:
    disp = np.array(Image.open(filename))
    disp = disp.astype(np.float32) / 100.0
    valid = disp > 0.0
    return disp, valid


# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md.
def readDispTartanAir(filename: str) -> Tuple[np.ndarray]:
    depth = np.load(filename)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid


# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py.
def readDispSintelStereo(filename: str) -> Tuple[np.ndarray]:
    a = np.array(Image.open(filename))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14))[..., 0]
    mask = np.array(Image.open(filename.replace("disparities", "occlusions")))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid


# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt.
def readDispFallingThings(filename: str) -> Tuple[np.ndarray]:
    a = np.array(Image.open(filename))
    with open("/".join(filename.split("/")[:-1] + ["_camera_settings.json"]), "r") as file:
        intrinsics = json.load(file)
    fx = intrinsics["camera_settings"][0]["intrinsic_settings"]["fx"]
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid


def readDispMiddlebury(filename: str) -> Tuple[np.ndarray]:
    ext = os.path.splitext(filename)[-1]
    if ext == ".png":
        disp = np.array(Image.open(filename)).astype(np.float32)
        valid = disp > 0.0
        return disp, valid
    elif os.path.basename(filename) == "disp0GT.pfm":
        disp = readPFM(filename).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = filename.replace("disp0GT.pfm", "mask0nocc.png")
        assert os.path.exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    else:
        disp = readPFM(filename).astype(np.float32)
        assert len(disp.shape) == 2
        valid = disp > 0.0
        return disp, valid


def writePFM(filename: str, array: np.ndarray) -> None:
    assert type(filename) is str and type(array) is np.ndarray and os.path.splitext(filename)[1] == ".pfm"
    with open(filename, "wb") as file:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            file.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        file.write(array.tobytes())


def readGen(filename: str, pil: bool=False) -> Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]:
    ext = os.path.splitext(filename)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(filename)
    elif ext == ".bin" or ext == ".raw":
        return np.load(filename)
    elif ext == ".flo":
        return readFlow(filename).astype(np.float32)
    elif ext == ".pfm":
        flow = readPFM(filename).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


def readFlow(filename: str) -> Union[None, np.ndarray]:
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy.

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(filename, "rb") as file:
        magic = np.fromfile(file, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file.")
            return None
        else:
            w = np.fromfile(file, np.int32, count=1)
            h = np.fromfile(file, np.int32, count=1)
            data = np.fromfile(file, np.float32, count=2 * int(w) * int(h))
            return np.resize(data, (int(h), int(w), 2))


def readPFM(filename: str) -> np.ndarray:
    file = open(filename, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")
    
    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")
    
    scale = float(file.readline().rstrip())
    if scale < 0: # Little-endian.
        endian = "<"
        scale = -scale
    else:
        endian = ">" # Big-endian.
    
    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename: str, uv: np.ndarray, v: np.ndarray=None) -> None:
    """
    Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels, stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv
    
    assert u.shape == v.shape
    height, width = u.shape
    file = open(filename, "wb")
    # Write the header.
    file.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(file)
    np.array(height).astype(np.int32).tofile(file)
    # Arrange into matrix form.
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(file)
    file.close()
