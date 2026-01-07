import pathlib as pth
import numpy as np

import random
from typing import Union, OrderedDict, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

import os
import sys
neural_net_dir = os.path.dirname(pth.Path(__file__).parent)
sys.path.append(neural_net_dir)


from utils.pcd_manipulation import rotate_points, tilt_points, transform_points, add_gaussian_noise
from utils.data_augmentation import cloud2sideViews_torch, gaussian_blur



class Dataset(IterableDataset):

    def __init__(self,
                 path_dir: Union[str, pth.Path],
                 resolution_xy: int,
                 num_classes: int,
                 batch_size: int,
                 weights: torch.Tensor = None,
                 shuffle: bool = True,
                 training: bool = True,
                 buffer: int = 250,
                 device: Optional[torch.device] = torch.device('cpu')):

        super(Dataset).__init__()

        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.device = device


        self.buffer_size = buffer
        self.weights = None
        if weights is not None:
            self.weights = weights.cpu().numpy()
            self.weights = self.weights*10
            self.weights = self.weights.astype(np.int32).clip(min=1)

            self.weights_dict = OrderedDict()
            for i, weight in enumerate(self.weights):
                self.weights_dict[i] = weight.item()
            


    def _key_streamer(self):
        """
        Generator over all keys. Each worker processes all keys,
        but only its assigned chunks within each key.
        """

        path_list = list(self.path.rglob('*.npy'))
        if self.shuffle:
            random.shuffle(path_list)

        worker_info = get_worker_info()
        if worker_info is None:
            iter_paths = path_list
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_paths = path_list[worker_id::total_workers]

        worker_buffer = []
        for path in iter_paths:

            label = int(path.stem.rsplit('_', 1)[-1])
                        
            if self.training: # TODO change this logic after next data processing
                if label == 0:
                    continue
            else:
                if label == 0:
                    label = self.num_classes
            label -= 1

            if self.shuffle is not None and self.weights is not None:
                for _ in range(self.weights_dict[label]):
                    worker_buffer.append((path, label))

                    if len(worker_buffer) >= self.buffer_size:
                        random.shuffle(worker_buffer)
                        for item in worker_buffer:
                            yield item
                            worker_buffer.pop(worker_buffer.index(item))

                    if len(worker_buffer) > 0:
                        random.shuffle(worker_buffer)
                        for item in worker_buffer:
                            yield item
                            worker_buffer.pop(worker_buffer.index(item))
            else:
                yield (path, label)


    def _process_cloud(self):
        stream = self._key_streamer()
        for (path, label) in stream:
            
            points = np.load(path)
            label = torch.asarray(label).long()

            cloud_tensor = torch.from_numpy(points[:, :3]).float()

            cloud_tensor = cloud_tensor.to(self.device)
            if self.shuffle:
                cloud_tensor = add_gaussian_noise(cloud_tensor, std=0.05)
                cloud_tensor = transform_points(cloud_tensor, device=self.device)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor, max_x_tilt_degrees=10, max_y_tilt_degrees=10, device=self.device)

            cloud_tensor = cloud2sideViews_torch(points=cloud_tensor, resolution_xy=self.resolution_xy)

            if self.shuffle:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.5, 0.8)
            else:
                kernel_size = 3
                sigma = 0.5

            cloud_tensor = gaussian_blur(cloud_tensor, kernel_size=(kernel_size, kernel_size), sigma=sigma)

            cloud_tensor = cloud_tensor.cpu()

            yield cloud_tensor, label

    def __iter__(self):
        stream = self._process_cloud()

        cloud_batch = []
        label_batch = []

        for (cloud, label) in stream:

            cloud_batch.append(cloud.unsqueeze(0))
            label_batch.append(label)

            if len(label_batch) >= self.batch_size:



                cloud_batch = torch.vstack(cloud_batch).float()
                label_batch = torch.asarray(label_batch).long()

                # print('BATCH', cloud_batch.shape)
                yield cloud_batch, label_batch

                cloud_batch = []
                label_batch = []

        if len(label_batch) > 0:
            cloud_batch = torch.vstack(cloud_batch).float()
            label_batch = torch.asarray(label_batch).long()

            yield cloud_batch, label_batch
