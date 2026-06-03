from __future__ import annotations

import pathlib as pth
import numpy as np
import h5py

import random
from typing import Union, OrderedDict, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

import os
import sys
neural_net_dir = os.path.dirname(pth.Path(__file__).parent)
sys.path.append(neural_net_dir)

from utils.pcd_manipulation import rotate_points, tilt_points, transform_points, add_gaussian_noise
from utils.data_augmentation import cloud2sideViews_torch

#########################################################
###################### ITERABLE #########################
#########################################################

class Dataset(IterableDataset):

    def __init__(self,
                 path_dir: Union[str, pth.Path],
                 resolution_xy: int,
                 num_classes: int,
                 batch_size: int,
                 weights: torch.Tensor = None,
                 shuffle: bool = True,
                 buffer: int = 250,
                 device: Optional[torch.device] = torch.device('cpu')):

        super(Dataset).__init__()

        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.buffer_size = buffer
        self.weights = None

        if weights is not None:
            self.weights = weights.cpu().numpy()
            self.weights = self.weights * 10
            self.weights = self.weights.astype(np.int32).clip(min=1)
            self.weights_dict = OrderedDict()
            for i, weight in enumerate(self.weights):
                self.weights_dict[i] = weight.item()

    def _iter_samples(self):
        """Yields (xyz: np.ndarray shape (N,3), label: int) from both .npy and .h5 files."""

        npy_paths = list(self.path.rglob('*.npy'))
        h5_paths  = list(self.path.rglob('*.h5'))

        worker_info = get_worker_info()
        if worker_info is not None:
            npy_paths = npy_paths[worker_info.id::worker_info.num_workers]
            h5_paths  = h5_paths[worker_info.id::worker_info.num_workers]

        if self.shuffle:
            random.shuffle(npy_paths)

        for path in npy_paths:
            label = int(path.stem.rsplit('_', 1)[-1])
            if label >= self.num_classes:
                continue
            arr   = np.load(path)
            yield arr[:, :3], label

        if self.shuffle:
            random.shuffle(h5_paths)

        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                if self.shuffle:
                    random.shuffle(keys)
                for key in keys:
                    chunk = f[key][:]           # (B, N, 4)
                    indices = list(range(chunk.shape[0]))
                    if self.shuffle:
                        random.shuffle(indices)
                    for i in indices:
                        row   = chunk[i]        # (N, 4)
                        label = int(row[0, 3])  # all points in this tree share the same label
                        yield row[:, :3], label

    def _key_streamer(self):
        worker_buffer = []

        for xyz, label in self._iter_samples():
            if self.shuffle and self.weights is not None:
                repeat = self.weights_dict.get(label, 1)
                for _ in range(repeat):
                    worker_buffer.append((xyz, label))

                if len(worker_buffer) >= self.buffer_size:
                    random.shuffle(worker_buffer)
                    for item in worker_buffer:
                        yield item
                    worker_buffer.clear()
            else:
                yield (xyz, label)

        if worker_buffer:
            random.shuffle(worker_buffer)
            for item in worker_buffer:
                yield item
            worker_buffer.clear()

    def _process_cloud(self):
        for xyz, label in self._key_streamer():
            label = torch.tensor(label).long()

            cloud_tensor = torch.from_numpy(xyz).float().to(self.device)

            if self.shuffle:
                cloud_tensor = add_gaussian_noise(cloud_tensor, std=0.05)
                cloud_tensor = transform_points(cloud_tensor, device=self.device)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor, max_x_tilt_degrees=10, max_y_tilt_degrees=10, device=self.device)

            cloud_tensor = cloud2sideViews_torch(points=cloud_tensor, resolution_xy=self.resolution_xy)

            yield cloud_tensor.cpu(), label

    def __iter__(self):
        cloud_batch = []
        label_batch = []

        for cloud, label in self._process_cloud():
            cloud_batch.append(cloud.unsqueeze(0))
            label_batch.append(label)

            if len(label_batch) >= self.batch_size:
                yield torch.vstack(cloud_batch).float(), torch.stack(label_batch).long()
                cloud_batch, label_batch = [], []

        if label_batch:
            yield torch.vstack(cloud_batch).float(), torch.stack(label_batch).long()


#########################################################
####################### CLASSIC #########################
#########################################################


class NpyDataset(torch.utils.data.Dataset):
 
    def __init__(self,
                 path_dir: Union[str, pth.Path],
                 resolution_xy: int = 350,
                 training: bool = True,
                 ignore_index: Optional[int] = None,
                 device: Optional[torch.device] = torch.device('cpu')):
 
        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.training = training
        self.device = device
        self.files = sorted(self.path.rglob('*.npy'))

        if ignore_index is not None:
            self.files = [f for f in self.files if int(f.stem.rsplit('_', 1)[-1]) < ignore_index]
        else:
            self.files = sorted(self.path.rglob('*.npy'))
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        path  = self.files[idx]
        label = int(path.stem.rsplit('_', 1)[-1])
    
        xyz   = np.load(path)[:, :3]

        # continue if height<1.5m
        if xyz[:, 2].max() - xyz[:, 2].min() < 1.5:
            continue
 
        cloud = torch.from_numpy(xyz).float().to(self.device)
        label = torch.tensor(label).long()
 
        if self.training:
            cloud = add_gaussian_noise(cloud, std=0.02)
            cloud = transform_points(cloud, device=self.device)
            cloud = rotate_points(cloud, device=self.device)
            cloud = tilt_points(cloud, max_x_tilt_degrees=15, max_y_tilt_degrees=15, device=self.device)

        cloud = cloud2sideViews_torch(cloud, resolution_xy=self.resolution_xy)

        return cloud.cpu(), label
    



import pathlib as pth
from typing import Optional, Union

import numpy as np
import torch

import pathlib as pth
from typing import Optional, Union

import numpy as np
import torch


def _rand_bool(p: float, device: torch.device) -> bool:
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(torch.rand((), device=device) < p)


def _bbox(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values
    span = torch.clamp(max_xyz - min_xyz, min=1e-6)
    return min_xyz, max_xyz, span


def _resample_points(points: torch.Tensor, n_points: int = 0) -> torch.Tensor:
    """
    If n_points <= 0, leave point count unchanged.

    If n_points > 0:
    - downsample when cloud has more points
    - oversample with replacement when cloud has fewer points
    - return unchanged when equal
    """
    if n_points <= 0:
        return points

    n = points.shape[0]

    if n <= 0:
        raise ValueError("Cannot resample empty point cloud")

    if n == n_points:
        return points

    device = points.device

    if n > n_points:
        idx = torch.randperm(n, device=device)[:n_points]
        return points[idx]

    extra = n_points - n
    idx = torch.randint(0, n, (extra,), device=device)

    return torch.cat([points, points[idx]], dim=0)


def _safe_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point cloud [N, >=3], got {tuple(points.shape)}")

    points = points[:, :3]
    mask = torch.isfinite(points).all(dim=1)
    points = points[mask]

    if points.shape[0] < 16:
        raise ValueError(f"Too few valid points: {points.shape[0]}")

    return points


def _normalize_for_masks(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_xyz, _, span = _bbox(points)
    norm = (points - min_xyz) / span
    return norm, min_xyz, span


def random_anisotropic_scale(
    points: torch.Tensor,
    xy_range: tuple[float, float] = (0.80, 1.35),
    z_range: tuple[float, float] = (0.75, 1.45),
) -> torch.Tensor:
    device = points.device
    center = points.mean(dim=0, keepdim=True)

    sx = torch.empty((), device=device).uniform_(*xy_range)
    sy = torch.empty((), device=device).uniform_(*xy_range)
    sz = torch.empty((), device=device).uniform_(*z_range)

    scale = torch.stack([sx, sy, sz]).view(1, 3)
    return (points - center) * scale + center


def random_nonuniform_thinning(
    points: torch.Tensor,
    min_keep: float = 0.35,
    max_keep: float = 0.85,
) -> torch.Tensor:
    """
    Simulates non-uniform LiDAR / branch visibility loss.
    This is intentionally not uniform point dropout.
    """
    device = points.device
    norm, _, _ = _normalize_for_masks(points)

    z = norm[:, 2]
    xy = norm[:, :2] - 0.5
    radial = torch.linalg.norm(xy, dim=1)

    base_keep = torch.empty((), device=device).uniform_(min_keep, max_keep)

    # Peripheral branches are more likely to disappear.
    peripheral_penalty = torch.clamp((radial - 0.25) / 0.45, 0.0, 1.0)

    # Random vertical bias: sometimes bottom/middle/top is sparser.
    z_center = torch.empty((), device=device).uniform_(0.15, 0.85)
    z_width = torch.empty((), device=device).uniform_(0.10, 0.30)
    z_penalty = torch.exp(-0.5 * ((z - z_center) / z_width) ** 2)

    keep_prob = base_keep
    keep_prob = keep_prob - 0.35 * peripheral_penalty
    keep_prob = keep_prob - 0.25 * z_penalty
    keep_prob = torch.clamp(keep_prob, 0.08, 1.0)

    keep = torch.rand(points.shape[0], device=device) < keep_prob

    if keep.sum() < 64:
        return points

    return points[keep]


def random_local_cuboid_dropout(
    points: torch.Tensor,
    min_cuboids: int = 2,
    max_cuboids: int = 8,
    drop_prob_inside: tuple[float, float] = (0.45, 0.90),
) -> torch.Tensor:
    """
    Removes local blocks/patches. This approximates missing branches,
    occlusion, or local segmentation failure.
    """
    device = points.device
    norm, _, _ = _normalize_for_masks(points)

    keep = torch.ones(points.shape[0], dtype=torch.bool, device=device)
    n_cuboids = int(torch.randint(min_cuboids, max_cuboids + 1, (), device=device))

    for _ in range(n_cuboids):
        center = torch.rand(3, device=device)

        # Small XY/Z boxes. Z can be larger to mimic vertical missing regions.
        size = torch.empty(3, device=device)
        size[0].uniform_(0.08, 0.24)
        size[1].uniform_(0.08, 0.24)
        size[2].uniform_(0.08, 0.30)

        low = center - size / 2.0
        high = center + size / 2.0

        inside = ((norm >= low) & (norm <= high)).all(dim=1)

        p_drop = torch.empty((), device=device).uniform_(*drop_prob_inside)
        drop = inside & (torch.rand(points.shape[0], device=device) < p_drop)
        keep &= ~drop

    if keep.sum() < 64:
        return points

    return points[keep]


def random_height_band_dropout(
    points: torch.Tensor,
    max_bands: int = 3,
    band_height_range: tuple[float, float] = (0.04, 0.16),
    drop_prob_inside: tuple[float, float] = (0.40, 0.85),
) -> torch.Tensor:
    """
    Simulates broken crowns/stems and missing horizontal LiDAR slices.
    """
    device = points.device
    norm, _, _ = _normalize_for_masks(points)
    z = norm[:, 2]

    keep = torch.ones(points.shape[0], dtype=torch.bool, device=device)
    n_bands = int(torch.randint(1, max_bands + 1, (), device=device))

    for _ in range(n_bands):
        center = torch.empty((), device=device).uniform_(0.05, 0.95)
        height = torch.empty((), device=device).uniform_(*band_height_range)
        inside = torch.abs(z - center) < height / 2.0

        p_drop = torch.empty((), device=device).uniform_(*drop_prob_inside)
        drop = inside & (torch.rand(points.shape[0], device=device) < p_drop)
        keep &= ~drop

    if keep.sum() < 64:
        return points

    return points[keep]


def random_vertical_crop(
    points: torch.Tensor,
    max_crop_ratio: float = 0.18,
) -> torch.Tensor:
    """
    Simulates bad top/bottom segmentation.
    Sometimes removes bottom, sometimes top, sometimes both.
    """
    device = points.device
    norm, _, _ = _normalize_for_masks(points)
    z = norm[:, 2]

    crop_bottom = _rand_bool(0.60, device)
    crop_top = _rand_bool(0.50, device)

    bottom = 0.0
    top = 1.0

    if crop_bottom:
        bottom = float(torch.empty((), device=device).uniform_(0.00, max_crop_ratio))
    if crop_top:
        top = 1.0 - float(torch.empty((), device=device).uniform_(0.00, max_crop_ratio))

    keep = (z >= bottom) & (z <= top)

    if keep.sum() < 64:
        return points

    return points[keep]


def random_neighbor_stem_fragment(
    points: torch.Tensor,
    max_fraction: float = 0.12,
) -> torch.Tensor:
    """
    Adds a thin vertical fragment near the tree, mimicking neighboring-tree leakage.
    """
    device = points.device
    min_xyz, max_xyz, span = _bbox(points)
    n = points.shape[0]

    fragment_n = int(torch.randint(
        max(32, int(0.02 * n)),
        max(33, int(max_fraction * n)) + 1,
        (),
        device=device,
    ))

    z_min = min_xyz[2]
    z_max = max_xyz[2]
    z = torch.empty(fragment_n, device=device).uniform_(float(z_min), float(z_max))

    side = -1.0 if _rand_bool(0.5, device) else 1.0
    offset_x = side * torch.empty((), device=device).uniform_(0.35, 0.80) * span[0]
    offset_y = torch.empty((), device=device).uniform_(-0.35, 0.35) * span[1]

    base_xy = points[:, :2].mean(dim=0)
    x = base_xy[0] + offset_x + torch.randn(fragment_n, device=device) * span[0] * 0.015
    y = base_xy[1] + offset_y + torch.randn(fragment_n, device=device) * span[1] * 0.015

    fragment = torch.stack([x, y, z], dim=1)
    return torch.cat([points, fragment], dim=0)


def random_crown_leakage_fragment(
    points: torch.Tensor,
    max_fraction: float = 0.20,
) -> torch.Tensor:
    """
    Copies a crown-like subset and translates it sideways.
    This is a cheap but effective simulation of neighboring crown leakage.
    """
    device = points.device
    norm, _, span = _normalize_for_masks(points)

    z = norm[:, 2]
    crown_mask = z > torch.empty((), device=device).uniform_(0.45, 0.70)

    crown_idx = torch.nonzero(crown_mask, as_tuple=False).flatten()
    if crown_idx.numel() < 32:
        return points

    n = points.shape[0]
    fragment_n = int(torch.randint(
        max(32, int(0.03 * n)),
        max(33, int(max_fraction * n)) + 1,
        (),
        device=device,
    ))

    sample_idx = crown_idx[torch.randint(0, crown_idx.numel(), (fragment_n,), device=device)]
    fragment = points[sample_idx].clone()

    side = -1.0 if _rand_bool(0.5, device) else 1.0
    translation = torch.zeros(3, device=device)
    translation[0] = side * torch.empty((), device=device).uniform_(0.25, 0.75) * span[0]
    translation[1] = torch.empty((), device=device).uniform_(-0.35, 0.35) * span[1]
    translation[2] = torch.empty((), device=device).uniform_(-0.08, 0.08) * span[2]

    fragment = fragment + translation

    # Make leaked fragment sparse.
    keep = torch.rand(fragment.shape[0], device=device) < torch.empty((), device=device).uniform_(0.25, 0.70)
    fragment = fragment[keep]

    if fragment.shape[0] == 0:
        return points

    return torch.cat([points, fragment], dim=0)


def random_sparse_outlier_clusters(
    points: torch.Tensor,
    max_clusters: int = 4,
    max_fraction: float = 0.06,
) -> torch.Tensor:
    """
    Adds small sparse blobs/fragments around the cloud.
    """
    device = points.device
    min_xyz, max_xyz, span = _bbox(points)
    center = (min_xyz + max_xyz) / 2.0

    n = points.shape[0]
    total_n = int(torch.randint(
        max(16, int(0.01 * n)),
        max(17, int(max_fraction * n)) + 1,
        (),
        device=device,
    ))

    n_clusters = int(torch.randint(1, max_clusters + 1, (), device=device))
    pieces = []

    for _ in range(n_clusters):
        cluster_n = max(4, total_n // n_clusters)

        offset = torch.empty(3, device=device).uniform_(-0.75, 0.75) * span
        offset[2] = torch.empty((), device=device).uniform_(-0.10, 0.90) * span[2]

        cluster_center = center + offset
        sigma = torch.empty(3, device=device)
        sigma[0].uniform_(0.01, 0.08)
        sigma[1].uniform_(0.01, 0.08)
        sigma[2].uniform_(0.01, 0.12)
        sigma = sigma * span

        cluster = cluster_center.view(1, 3) + torch.randn(cluster_n, 3, device=device) * sigma.view(1, 3)
        pieces.append(cluster)

    return torch.cat([points, *pieces], dim=0)


def random_branch_emphasis_loss(
    points: torch.Tensor,
    keep_core_ratio: float = 0.90,
    keep_periphery_ratio: float = 0.35,
) -> torch.Tensor:
    """
    Preferentially removes peripheral points. This approximates weak branch visibility.
    """
    device = points.device
    norm, _, _ = _normalize_for_masks(points)

    xy = norm[:, :2] - 0.5
    radial = torch.linalg.norm(xy, dim=1)
    z = norm[:, 2]

    # Crown/branch area: more peripheral and upper parts.
    branch_score = 0.65 * torch.clamp((radial - 0.15) / 0.45, 0.0, 1.0)
    branch_score += 0.35 * torch.clamp((z - 0.25) / 0.70, 0.0, 1.0)
    branch_score = torch.clamp(branch_score, 0.0, 1.0)

    keep_prob = keep_core_ratio * (1.0 - branch_score) + keep_periphery_ratio * branch_score
    keep = torch.rand(points.shape[0], device=device) < keep_prob

    if keep.sum() < 64:
        return points

    return points[keep]


def grajewo_domain_augment(
    points: torch.Tensor,
    n_points: int = 0,
) -> torch.Tensor:
    """
    On-the-fly domain augmentation for original -> real segmented forest LiDAR.

    Point count may temporarily change.
    If n_points > 0, returns exactly n_points.
    If n_points <= 0, returns augmented cloud with natural augmented size.
    """
    points = _safe_points(points)

    device = points.device

    if _rand_bool(0.35, device):
        points = random_anisotropic_scale(points)

    if _rand_bool(0.70, device):
        points = random_nonuniform_thinning(points)

    if _rand_bool(0.45, device):
        points = random_branch_emphasis_loss(points)

    if _rand_bool(0.45, device):
        points = random_local_cuboid_dropout(points)

    if _rand_bool(0.30, device):
        points = random_height_band_dropout(points)

    if _rand_bool(0.25, device):
        points = random_vertical_crop(points)

    if _rand_bool(0.30, device):
        points = random_crown_leakage_fragment(points)

    if _rand_bool(0.18, device):
        points = random_neighbor_stem_fragment(points)

    if _rand_bool(0.20, device):
        points = random_sparse_outlier_clusters(points)

    points = _safe_points(points)
    points = _resample_points(points, n_points=n_points)

    return points
    
class NpyDatasetAug(torch.utils.data.Dataset):
    def __init__(
        self,
        path_dir: Union[str, pth.Path],
        resolution_xy: int = 350,
        training: bool = True,
        ignore_index: Optional[int] = None,
        device: Optional[torch.device] = torch.device("cpu"),
        n_points: int = 0,
        use_domain_aug: bool = True,
    ):
        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.training = training
        self.device = device if device is not None else torch.device("cpu")
        self.n_points = n_points
        self.use_domain_aug = use_domain_aug
        self.ignore_index = ignore_index

        self.files = sorted(self.path.rglob("*.npy"))

        if ignore_index is not None:
            kept_files = []

            for file_path in self.files:
                label = int(file_path.stem.rsplit("_", 1)[-1])

                if label >= ignore_index:
                    continue

                kept_files.append(file_path)

            self.files = kept_files

        if not self.files:
            raise RuntimeError(f"No .npy files found under: {self.path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label_value = int(path.stem.rsplit("_", 1)[-1])

        if self.ignore_index is not None and label_value >= self.ignore_index:
            raise RuntimeError(
                f"Dataset filtering failed: label={label_value}, "
                f"ignore_index={self.ignore_index}, file={path}"
            )

        xyz = np.load(path, allow_pickle=False)[:, :3]




        cloud = torch.from_numpy(xyz).float().to(self.device)
        cloud = _safe_points(cloud)

        # Optional fixed-size input before augmentation.
        # If n_points == 0, this does nothing.
        cloud = _resample_points(cloud, n_points=self.n_points)

        label = torch.tensor(label_value).long()

        if self.training:
            cloud = add_gaussian_noise(cloud, std=0.02)
            cloud = transform_points(cloud, device=self.device)
            cloud = rotate_points(cloud, device=self.device)
            cloud = tilt_points(
                cloud,
                max_x_tilt_degrees=15,
                max_y_tilt_degrees=15,
                device=self.device,
            )

            if self.use_domain_aug:
                cloud = grajewo_domain_augment(
                    cloud,
                    n_points=self.n_points,
                )

        # Final enforcement. If n_points == 0, this does nothing.
        cloud = _resample_points(cloud, n_points=self.n_points)

        cloud = cloud2sideViews_torch(
            cloud,
            resolution_xy=self.resolution_xy,
        )

        return cloud.cpu(), label