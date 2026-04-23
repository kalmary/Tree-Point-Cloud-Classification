import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json
from typing import Union, Dict


class ResNetClassifier(nn.Module):
    """
    Configurable ResNet classifier trained from scratch.
    
    Args:
        config: Dictionary containing model configuration
        num_classes: Number of output classes
    
    Config structure:
    {
        "model_version": "resnet50",  # resnet18, resnet34, resnet50, resnet101, resnet152
        "in_channels": 5,
        "dropout": 0.4
    }
    """
    
    MODEL_REGISTRY = {
        'resnet18':  (models.resnet18,  512),
        'resnet34':  (models.resnet34,  512),
        'resnet50':  (models.resnet50,  2048),
        'resnet101': (models.resnet101, 2048),
        'resnet152': (models.resnet152, 2048),
    }
    
    def __init__(self, config: Dict, num_classes: int = 18):
        super().__init__()
        
        self.model_version = config.get('model_version', 'resnet50')
        self.in_channels   = config.get('in_channels', 5)
        self.dropout       = config.get('dropout', 0.4)
        self.num_classes   = num_classes
        
        if self.model_version not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Model version '{self.model_version}' not supported. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        self._build_model()
        self._modify_input_layer()
        self._replace_classifier()
    
    def _build_model(self):
        model_fn, _ = self.MODEL_REGISTRY[self.model_version]
        self.model = model_fn(weights=None)
    
    def _modify_input_layer(self):
        """Replace conv1 to accept in_channels instead of 3."""
        old_conv = self.model.conv1
        
        if old_conv.in_channels == self.in_channels:
            return
        
        new_conv = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.model.conv1 = new_conv
    
    def _replace_classifier(self):
        """Replace fc head with dropout + linear."""
        _, in_features = self.MODEL_REGISTRY[self.model_version]
        
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, self.num_classes)
        )
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], num_classes: int = 18):
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config=config, num_classes=num_classes)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        """x: (batch, in_channels, H, W)"""
        return self.model(x)


if __name__ == "__main__":
    config = {
        "model_version": "resnet50",
        "in_channels": 5,
        "dropout": 0.4
    }
    
    model = ResNetClassifier(config=config, num_classes=18)
    
    batch = torch.randn(2, 5, 350, 350)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_total_params():,}")

    print("\n" + "="*50)
    for version in ResNetClassifier.MODEL_REGISTRY:
        m = ResNetClassifier({"model_version": version, "in_channels": 5, "dropout": 0.4}, num_classes=18)
        print(f"{version:12s}: {m.get_total_params():,} parameters")