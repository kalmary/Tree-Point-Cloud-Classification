import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json
from typing import Union, Dict


class EfficientNetClassifier(nn.Module):
    """
    Configurable EfficientNet classifier trained from scratch.
    
    Args:
        config: Dictionary containing model configuration
        num_classes: Number of output classes
    
    Config structure:
    {
        "model_version": "b4",  # b0, b1, b2, b3, b4, b5, b6, b7
        "in_channels": 5,
        "dropout": 0.4
    }
    """
    
    # Mapping of model versions to their constructors
    MODEL_REGISTRY = {
        'b0': models.efficientnet_b0,
        'b1': models.efficientnet_b1,
        'b2': models.efficientnet_b2,
        'b3': models.efficientnet_b3,
        'b4': models.efficientnet_b4,
        'b5': models.efficientnet_b5,
        'b6': models.efficientnet_b6,
        'b7': models.efficientnet_b7,
    }
    
    def __init__(self, config: Dict, num_classes: int = 18):
        super().__init__()
        
        # Extract config parameters
        self.model_version = config.get('model_version', 'b4')
        self.in_channels = config.get('in_channels', 5)
        self.dropout = config.get('dropout', 0.4)
        self.num_classes = num_classes
        
        # Validate model version
        if self.model_version not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Model version '{self.model_version}' not supported. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        # Build model from scratch
        self._build_model()
        
        # Modify input channels if needed
        self._modify_input_layer()
        
        # Replace classifier head
        self._replace_classifier()
    
    def _build_model(self):
        """Load the EfficientNet architecture without pretrained weights."""
        model_fn = self.MODEL_REGISTRY[self.model_version]
        self.model = model_fn(weights=None)
    
    def _modify_input_layer(self):
        """
        Modify first convolutional layer to accept custom number of channels.
        Randomly initialize with Kaiming initialization.
        """
        old_conv = self.model.features[0][0]
        
        if old_conv.in_channels == self.in_channels:
            return
        
        # Create new conv layer with desired input channels
        new_conv = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Initialize with Kaiming/He initialization
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Replace the layer
        self.model.features[0][0] = new_conv
    
    def _replace_classifier(self):
        """Replace the classification head with custom output size."""
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], num_classes: int = 18):
        """
        Load model from a JSON config file.
        
        Args:
            config_path: Path to JSON config file
            num_classes: Number of output classes
            
        Returns:
            EfficientNetClassifier instance
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(config=config, num_classes=num_classes)
    
    def get_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, in_channels, H, W)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        return self.model(x)


# Example usage
if __name__ == "__main__":
    # Example 1: Using dict config
    config = {
        "model_version": "b4",
        "in_channels": 5,
        "dropout": 0.4
    }
    
    model = EfficientNetClassifier(config=config, num_classes=18)
    
    # Test forward pass
    batch = torch.randn(2, 5, 350, 350)
    output = model(batch)
    print(f"\nOutput shape: {output.shape}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Example 2: Loading from JSON file
    # model = EfficientNetClassifier.from_config_file('config.json', num_classes=18)
    
    # Example 3: Different model sizes
    print("\n" + "="*50)
    
    configs = {
        "b0": {"model_version": "b0", "in_channels": 5, "dropout": 0.3},
        "b2": {"model_version": "b2", "in_channels": 5, "dropout": 0.35},
        "b4": {"model_version": "b4", "in_channels": 5, "dropout": 0.4},
    }
    
    for name, cfg in configs.items():
        m = EfficientNetClassifier(cfg, num_classes=18)
        print(f"{name}: {m.get_total_params():,} parameters")