import torch
import torch.nn as nn
import pathlib as pth

from src.utils import data_augmentation, pcd_manipulation


class TreeClassifier:
    def __init__(model_name: str = None,
                 config_dir: Union[str, pth.Path] = "./final_files",
                 device: torch.device = torch.device('cpu')):