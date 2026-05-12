import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json
from typing import Union, Dict, Optional


class EfficientNetClassifier(nn.Module):
    """
    Configurable EfficientNet classifier with pretrained ImageNet weights.
    Pretrained conv1 weights are averaged across channels to support non-RGB inputs.

    Args:
        config: Dictionary containing model configuration
        num_classes: Number of output classes

    Config structure:
    {
        "model_version": "efficientnet_b2",  # b0, b1, b2, b3, b4
        "in_channels": 5,
        "dropout": 0.4
    }
    """

    MODEL_REGISTRY = {
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
        'efficientnet_b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT, 1280),
        'efficientnet_b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, 1408),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, 1536),
        'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792),
    }

    def __init__(self, config: Dict, num_classes: int = 18):
        super().__init__()

        self.model_version = config.get('model_version', 'efficientnet_b2')
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
        model_fn, weights, _ = self.MODEL_REGISTRY[self.model_version]
        self.model = model_fn(weights=weights)

    def _modify_input_layer(self):
        old_conv = self.model.features[0][0]

        if old_conv.in_channels == self.in_channels:
            return

        old_w = old_conv.weight.data  # (C_out, 3, kH, kW)

        new_conv = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        new_conv.weight.data[:, :3, :, :] = old_w
        self.model.features[0][0] = new_conv

    def _replace_classifier(self):
        """Replace classifier head with dropout + linear."""
        _, _, in_features = self.MODEL_REGISTRY[self.model_version]

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, self.num_classes)
        )

    def freeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

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

    def forward(self, x, targets: Optional[torch.Tensor] = None):
        feat = self.model.features(x)
        feat = self.model.avgpool(feat)
        feat = feat.flatten(1)
        logits = self.model.classifier(feat)
        if targets is None:
            return logits
        return logits, feat

def test_model():
    # for version, (model_fn, weights, stored_val) in EfficientNetClassifier.MODEL_REGISTRY.items():
    #     m = model_fn(weights=None)
    #     actual = m.classifier[1].in_features
    #     print(f"{version}: stored={stored_val}, actual={actual}, match={stored_val == actual}")

    config = {
        "model_version": "efficientnet_b2",
        "in_channels": 5,
        "dropout": 0.4
    }

    model = EfficientNetClassifier(config=config, num_classes=15)

    batch = torch.randn(2, 5, 350, 350)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_total_params():,}")

    print("\n" + "="*50)
    for version in EfficientNetClassifier.MODEL_REGISTRY:
        m = EfficientNetClassifier({"model_version": version, "in_channels": 5, "dropout": 0.4}, num_classes=15)
        print(f"{version:20s}: {m.get_total_params():,} parameters")

if __name__ == "__main__":
    test_model()
