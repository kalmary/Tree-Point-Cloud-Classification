from typing import Union

from torchvision.transforms import GaussianBlur
import torch
import numpy as np


def gaussian_blur(img: Union[torch.Tensor, np.ndarray], kernel_size=(5, 5), sigma=1.5):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.float()
    # Create a Gaussian kernel
    gaussian_kernel = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    # Apply the Gaussian kernel to the image
    blurred_img = gaussian_kernel(img)

    return blurred_img

def cloud2sideViews_torch(points: torch.Tensor, resolution_xy: int, margin_ratio: float = 0.05) -> torch.Tensor:
    points = points.type(torch.float32)

    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values

    center = (min_xyz + max_xyz) / 2
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2 * (1 + 2 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half


    def to_grid(val, min_val, max_val):
        return torch.clamp(
            ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
            0, resolution_xy - 1
        )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    gx = to_grid(x, cube_min[0], cube_max[0])
    gy = to_grid(y, cube_min[1], cube_max[1])
    gz = to_grid(z, cube_min[2], cube_max[2])

    views = []

    def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
        y_idx, x_idx = indices_2d
        if flip_y:
            y_idx = resolution_xy - 1 - y_idx
        if flip_x:
            x_idx = resolution_xy - 1 - x_idx

        flat_indices = y_idx * resolution_xy + x_idx
        depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'), dtype=torch.float32,
                               device=distances.device)

        # Use scatter_reduce to keep the minimum distance per pixel
        depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances, reduce='amin', include_self=True)

        img = depth_map.view(resolution_xy, resolution_xy)
        img[img == float('inf')] = 0  # Replace untouched pixels

        # Normalize non-zero pixels to [0, 1]
        nonzero_mask = img > 0
        if torch.any(nonzero_mask):
            values = img[nonzero_mask]
            min_val = values.min()
            max_val = values.max()
            img[nonzero_mask] = (values - min_val) / (max_val - min_val + 1e-8)

        return img

    # Compute distance from each wall
    dist_top = cube_max[2] - z
    views.append(build_depth_map((gy, gx), dist_top))

    dist_front = cube_max[1] - y
    views.append(build_depth_map((gz, gx), dist_front, flip_y=True))

    dist_back = y - cube_min[1]
    views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

    dist_left = cube_max[0] - x
    views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

    dist_right = x - cube_min[0]
    views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

    return torch.stack(views, dim=0)
