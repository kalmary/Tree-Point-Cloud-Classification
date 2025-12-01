from typing import Optional
import pathlib as pth

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


from model_pipeline.RandLANet_CB import RandLANet
from utils import load_json, load_model, pcd_manipulation

class SegmentClass:
    def __init__(self,
                 voxel_size_big: Optional[float] = None,
                 overlap: float = 0.4,
                 device: torch.device = torch.device('cpu')):
        
        self.voxel_size_small = None
        self.voxel_size_big = voxel_size_big
        self.overlap = overlap

        self.device = device
        self._scaler = None
        self._model_config = None
        self._model = None


    # TODO adjust model loading 
    def _load_config(self, config_path: pth.Path) -> dict:
        config_dict = load_json(config_path)
        self._model_config: dict = config_dict['model_config']
        self.voxel_size_small: float = self.model_config['max_voxel_dim']

        return config_dict

    def _load_segmModel(self, path2model) -> nn.Module:
        model = RandLANet.from_config_file(self._model_config, self._model_config['num_classes'])
        self._model: nn.Module = load_model(file_path= path2model,
                                 model=model,
                                 device=self.device
                                 )
    

        return model
    
    def _init_scaler(self, data: np.ndarray) -> MinMaxScaler:
        self._scaler = MinMaxScaler()
        return self._scaler
    
    @property
    def model_config(self) -> dict:
        return self._model_config
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def scaler(self) -> MinMaxScaler:
        return self._scaler
    
    def _segment_small_voxel(self,
                             points: np.ndarray,
                             intensity: np.ndarray):


        voxel_all = np.full((points.shape[0], 3), np.nan, dtype = np.float32)
        voxel_probs_all = np.full((points.shape[0], self._model_config['num_classes']), np.nan, dtype=np.float32)

        checksum = 0

        for i, (voxel_idx, noise) in enumerate(pcd_manipulation.voxelGridFragmentation(points,
                                                                    voxel_size = np.array([35., 35.]),
                                                                    num_points = num_points,
                                                                    overlap_ratio=0.4)):

            voxel = points[voxel_idx]
            voxel0 = voxel.copy()

            voxel -= voxel.mean(axis= 0)

            intensity_voxel = intensity[voxel_idx]

            global_idx, voxel_idx = np.unique(voxel_idx, return_index=True) # global - unique z points, voxel - unique z voxel
            global_idx = np.sort(global_idx)

            voxel_idx = np.sort(voxel_idx)

            checksum += voxel_idx.shape[0]

            voxel = np.concatenate([voxel, intensity_voxel.reshape(-1, 1)], axis = 1)

            if not noise:

                voxel_probs = model_predict(modelSegm, voxel, mode = 0, device= device) # class from 2
                assert voxel_probs.shape[0] == voxel.shape[0]

            else:
                voxel_probs = np.zeros((voxel.shape[0], num_classes))
                voxel_probs[:, 0] = 1 # highest prob for class 0


            voxel = voxel0[voxel_idx] # remove reduntant points and overwrite centered voxel
            voxel_probs = voxel_probs[voxel_idx]

            voxel_all[global_idx] = voxel

            voxel_probs_all[global_idx] = voxel_probs
            voxel_all[global_idx] = voxel


            if pbar is not None:
                pbar.set_postfix({
                    'CURRENT PROCESS': f'Segmentation by small voxels: NUM POINTS: {checksum}, VOXEL: {i}'
                })

        mask0 = np.isnan(voxel_all).any(axis = 1)
        mask1 = np.isnan(voxel_probs_all).any(axis=1)

        voxel_all = voxel_all[~mask0]
        voxel_probs_all = voxel_probs_all[~mask1]

        del voxel_probs, voxel, voxel0, voxel_idx

        return voxel_all, voxel_probs_all
    
    

    def _segment_big_voxel(self, points: np.ndarray, intensity: np.ndarray):

        for indices in pcd_manipulation.voxelGridFragmentation(data=points,
                                                               num_points=self._model_config['num_points'],
                                                               voxel_size=np.array([self.voxel_size_big, self.voxel_size_big]),
                                                               overlap_ratio=0,
                                                               shuffle=False):
            
            points_chunk = points[indices]
            points_chunk -= points_chunk.mean(axis = 0)

            labels_full = np.zeros(intensity.shape, dtype=np.int32)
            idx_full = np.arange(intensity.shape[0], dtype=np.int32)

            intensity_chunk = intensity[indices]
            idx_chunk = idx_full[indices]
    
    def segment_pcd(self, points: np.ndarray, intensity: np.ndarray, fragment_pcd_threshold: int = 20*10e6) -> np.ndarray:

        intensity = self._scaler.fit_transform(intensity)
        points -= points.mean(axis = 0)

        num_points = points.shape[0]
        if num_points < fragment_pcd_threshold:
            self._segment_small_voxel(points, intensity)
        else:
            self._segment_big_voxel(points, intensity)

        






    


