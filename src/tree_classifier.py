import argparse
import pathlib as pth
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

if __package__:
    from .final_files.model_en import EfficientNetClassifier
    from .utils.data_augmentation import cloud2side_views_torch
    from .utils.nn_utils import load_json, load_model
else:
    from final_files.model_en import EfficientNetClassifier
    from utils.data_augmentation import cloud2side_views_torch
    from utils.nn_utils import load_json, load_model

class TreeClassifier:
    def __init__(self,
                 model_name: str = None,
                 config_dir: Union[str, pth.Path] = "src/final_files",
                 device: torch.device = torch.device('cpu')):
        self.model_name = model_name + '.pt'
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self._config = None
        self._model_config = None
        self._model = None
        self._load_config(config_dir)
        self._load_model(config_dir)

    # TODO adjust model loading 
    def _load_config(self, config_dir: Optional[Union[pth.Path, str]] = None) -> dict:

        config_path = pth.Path(config_dir).joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)

        self._config = config_dict
        self._model_config: dict = config_dict["model_config"]

    def _load_model(self, model_dir: Union[pth.Path, str] = "./final_files") -> nn.Module:

        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = EfficientNetClassifier(self._model_config, self._config['num_classes'])
        _model: nn.Module = load_model(file_path=path2model,
                                            model=model,
                                            device=self.device)
        _model.eval()
        self._model = _model
        
    def predict(self, cloud: np.ndarray):
        if cloud.shape[0] == 0:
            return np.array([15], dtype=np.int64)

        cloud = torch.as_tensor(cloud, dtype=torch.float64)
        imgs = cloud2side_views_torch(cloud, resolution_xy=350)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        imgs = imgs.to(self.device)      
        with torch.no_grad():             
            output = self._model(imgs)
        labels = output.argmax(dim=-1).cpu().detach().numpy()
        return labels.flatten()



def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Load a tree species classification model.")
    parser.add_argument("--model-name", default="ResNetTreeV0_61")
    parser.add_argument(
        "--config-dir",
        type=pth.Path,
        default=pth.Path(__file__).parent / "final_files",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)
    device_name = "cuda" if args.device == "gpu" else args.device
    TreeClassifier(
        model_name=args.model_name,
        config_dir=args.config_dir,
        device=torch.device(device_name),
    )


def test_predict_preserves_projection_and_label_selection(monkeypatch):
    classifier = object.__new__(TreeClassifier)
    classifier.device = torch.device("cpu")
    received = {}

    def project(cloud, resolution_xy):
        received["cloud"] = cloud.clone()
        received["resolution_xy"] = resolution_xy
        return torch.zeros((5, 4, 4), dtype=torch.float32)

    class Model:
        def __call__(self, batch):
            received["batch_shape"] = tuple(batch.shape)
            return torch.tensor([[0.1, 0.8, 0.1]])

    monkeypatch.setattr(
        __import__(__name__, fromlist=["cloud2side_views_torch"]),
        "cloud2side_views_torch",
        project,
    )
    classifier._model = Model()
    cloud = np.arange(12, dtype=np.float64).reshape(4, 3)

    result = classifier.predict(cloud)

    assert received["resolution_xy"] == 350
    assert received["cloud"].dtype == torch.float64
    assert received["batch_shape"] == (1, 5, 4, 4)
    assert np.array_equal(result, np.array([1]))


def test_predict_preserves_empty_cloud_label():
    classifier = object.__new__(TreeClassifier)

    assert np.array_equal(
        classifier.predict(np.empty((0, 3))),
        np.array([15], dtype=np.int64),
    )


if __name__ == '__main__':
    main()
