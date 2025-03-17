import open3d as o3d
import torch
from typing import Any
from utils.utils import coords_grid


def pc_generator(left_img: torch.Tensor, valid_gt: torch.Tensor, disp_gt: torch.Tensor, disp_pr: torch.Tensor, camera_info: Any) -> None:
    # Get the resolutions.
    _, h, w = left_img.shape

    # Decompress the camera information.
    focal = camera_info["focal"]
    baseline = camera_info["baseline"]
    intrinsic = camera_info["int"]

    # Convert the disparity into depth.
    depth_gt = ((focal * baseline) / disp_gt).masked_fill(valid_gt==0.0, 0.0).float()
    depth_pr = (focal * baseline) / disp_pr.detach()

    # Get the pixel grid.
    coords = coords_grid(1, h, w).squeeze(0).permute(1, 2, 0).reshape(-1, 2)
    ones = torch.ones_like(disp_gt).permute(1, 2, 0).reshape(-1, 1)
    
    # Get the point cloud.
    rgb = left_img.permute(1, 2, 0).reshape(-1, 3)
    xyz = torch.cat([coords, ones], dim=-1)
    xyz_gt = xyz * depth_gt.permute(1, 2, 0).reshape(-1, 1)
    xyz_gt = (torch.linalg.pinv(intrinsic) @ xyz_gt.permute(1, 0)).permute(1, 0)
    xyz_gt = torch.cat([xyz_gt, rgb], dim=-1)
    xyz_pr = xyz * depth_pr.permute(1, 2, 0).reshape(-1, 1)
    xyz_pr = (torch.linalg.pinv(intrinsic) @ xyz_pr.permute(1, 0)).permute(1, 0)
    xyz_pr = torch.cat([xyz_pr, rgb], dim=-1)

    # Save the point cloud.
    pcd_gt = o3d.geometry.PointCloud()
    points_gt = xyz_gt[:, :3].numpy()
    colors_gt = xyz_gt[:, 3:].numpy() / 255.0
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
    o3d.io.write_point_cloud("./pcd_gt.ply", pcd_gt, write_ascii=True)
    pcd_pr = o3d.geometry.PointCloud()
    points_pr = xyz_pr[:, :3].numpy()
    colors_pr = xyz_pr[:, 3:].numpy() / 255.0
    pcd_pr.points = o3d.utility.Vector3dVector(points_pr)
    pcd_pr.colors = o3d.utility.Vector3dVector(colors_pr)
    o3d.io.write_point_cloud("./pcd_pr.ply", pcd_pr, write_ascii=True)
