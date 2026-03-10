from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import pathlib as pth

from final_files.model_test import EfficientNetClassifier
from final_files.ResNet import ResNetTreeClassifier
from utils import load_model, load_json
from utils.data_augmentation import gaussian_blur, cloud2sideViews_torch


class TreeClassifier:
    def __init__(self,
                 model_name: str = None,
                 config_dir: Union[str, pth.Path] = "src/final_files",
                 device: torch.device = torch.device('cpu')):
        self.model_name = model_name + '.pt'
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self._model_config = self._load_config(config_dir)
        self._model = self._load_model(config_dir)

    # TODO adjust model loading 
    def _load_config(self, config_dir: Optional[Union[pth.Path, str]] = None) -> dict:

        config_path = pth.Path(config_dir).joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        # self._model_config: dict = config_dict['model_config']
        self._model_config: dict = config_dict

        

        return config_dict

    def _load_model(self, model_dir: Union[pth.Path, str] = "./final_files") -> nn.Module:

        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = ResNetTreeClassifier(self._model_config['num_channels'], self._model_config['num_classes'])
        self._model: nn.Module = load_model(file_path=path2model,
                                            model=model,
                                            device=self.device)
        
    def predict(self, cloud: np.ndarray):

        cloud  = torch.asarray(cloud)
        imgs = cloud2sideViews_torch(cloud,
                                    resolution_xy=350)
        # imgs = gaussian_blur(imgs,
        #                      kernel_size=(3, 3),
        #                      sigma = 0.6)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        output = self._model(imgs)
        
        labels = output.argmax(dim = -1).numpy()

        labels = labels.flatten()
        return labels

def main():
    TreeClassifier(model_name="ResNetTreeV0_61", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

if __name__ == '__main__':
    main()