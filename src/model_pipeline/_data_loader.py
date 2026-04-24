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
                 return_others: bool = False,
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
                 return_others: bool = False,
                 device: Optional[torch.device] = torch.device('cpu')):
 
        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.training = training
        self.device = device
        self.files = sorted(self.path.rglob('*.npy'))

        if not return_others:
            self.files = [f for f in self.files if int(f.stem.rsplit('_', 1)[-1]) < 15]
        else:
            self.files = sorted(self.path.rglob('*.npy'))
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        path  = self.files[idx]
        label = int(path.stem.rsplit('_', 1)[-1])
    
        xyz   = np.load(path)[:, :3]
 
        cloud = torch.from_numpy(xyz).float().to(self.device)
        label = torch.tensor(label).long()
 
        if self.training:
            cloud = add_gaussian_noise(cloud, std=0.07)
            cloud = transform_points(cloud, device=self.device)
            cloud = rotate_points(cloud, device=self.device)
            cloud = tilt_points(cloud, max_x_tilt_degrees=15, max_y_tilt_degrees=15, device=self.device)

        cloud = cloud2sideViews_torch(cloud, resolution_xy=self.resolution_xy)

        return cloud.cpu(), label