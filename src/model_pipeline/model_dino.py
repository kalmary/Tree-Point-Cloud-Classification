from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"xFormers is not available.*",
    category=UserWarning,
)

import json
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn


class DINOv2DepthmapClassifier(nn.Module):
    """
    Frozen DINOv2 classifier for multi-channel depthmap inputs.

    Input:
        Tensor shaped [B, in_channels, H, W]
        Expected in this project: [B, 5, 350, 350]

    Architecture:
        in_channels -> 3 via trainable 1x1 convolution
        ImageNet normalization
        frozen DINOv2 backbone
        two-layer MLP classifier

    Args:
        config:
            {
                "dino_version": "dinov2_vits14",  # or "dinov2_vitb14"
                "in_channels": 5,
                "dropout": 0.4
            }
        num_classes:
            Number of output classes.
    """

    MODEL_REGISTRY = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
    }

    def __init__(self, config: Dict, num_classes: int):
        super().__init__()

        self.dino_version = config.get("dino_version", "dinov2_vits14")
        self.in_channels = int(config.get("in_channels", 5))
        self.dropout = float(config.get("dropout", 0.4))
        self.num_classes = int(num_classes)

        if self.dino_version not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported DINOv2 version: {self.dino_version!r}. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = self.MODEL_REGISTRY[self.dino_version]

        self._build_input_adapter()
        self._build_backbone()
        self._build_classifier()
        self._register_normalization_buffers()

    def _build_input_adapter(self) -> None:
        """
        Learnable channel adapter:
            [B, in_channels, H, W] -> [B, 3, H, W]

        For five depthmap views, this learns the best RGB-like mixture
        for the frozen DINO backbone.
        """
        self.input_adapter = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        nn.init.kaiming_normal_(
            self.input_adapter.weight,
            mode="fan_out",
            nonlinearity="linear",
        )
        nn.init.zeros_(self.input_adapter.bias)

    def _build_backbone(self) -> None:
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self.dino_version,
        )

        self.backbone.eval()
        self.freeze_backbone()

    def _build_classifier(self) -> None:
        hidden_dim = max(self.feature_dim // 2, 128)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _register_normalization_buffers(self) -> None:
        """
        DINOv2 hub models are conventionally used with ImageNet normalization.
        Buffers move automatically with .to(device).
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        """
        Keep DINO in eval mode even when the classifier is training.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shaped [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        x = self.input_adapter(x)
        x = (x - self.imagenet_mean) / self.imagenet_std

        with torch.no_grad():
            features = self.backbone(x)

        logits = self.classifier(features)
        return logits

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        num_classes: int,
    ) -> "DINOv2DepthmapClassifier":
        config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        return cls(config=config, num_classes=num_classes)

    def get_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def get_total_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def test_model() -> None:
    config = {
        "dino_version": "dinov2_vits14",
        "in_channels": 5,
        "dropout": 0.4,
    }

    model = DINOv2DepthmapClassifier(
        config=config,
        num_classes=15,
    )

    batch = torch.randn(2, 5, 350, 350)
    logits = model(batch)

    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    print("\nModel variants:")
    for version in DINOv2DepthmapClassifier.MODEL_REGISTRY:
        cfg = {
            "dino_version": version,
            "in_channels": 5,
            "dropout": 0.4,
        }
        instance = DINOv2DepthmapClassifier(
            config=cfg,
            num_classes=15,
        )
        print(
            f"{version:18s} | "
            f"total={instance.get_total_params():,} | "
            f"trainable={instance.get_trainable_params():,}"
        )


if __name__ == "__main__":
    test_model()