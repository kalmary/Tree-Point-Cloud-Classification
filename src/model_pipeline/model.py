import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union
from pathlib import Path
import json



class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResBottleneckBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion=4, dropout=0.0, downsample=None):
        super().__init__()

        padding = kernel_size // 2

        # Narrow conv layer (1x1)
        self.conv1 = nn.Conv2d(in_channels, out_channels // expansion, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // expansion)

        # Main conv layer (3x3 or larger)
        self.conv2 = nn.Conv2d(out_channels // expansion, out_channels // expansion,
                               kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // expansion)

        # Wide conv layer (1x1)
        self.conv3 = nn.Conv2d(out_channels // expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNN2D_Residual(nn.Module):
    def __init__(self, config_data, num_classes):
        super(CNN2D_Residual, self).__init__()

        dropout = config_data['global_params']['dropout']

        # INITIAL CONV - for 5 input channels
        init_cfg = config_data['initial_conv']
        self.initial_conv = nn.Sequential(
            nn.Conv2d(config_data['in_channels'], init_cfg['out_channels'], 
                      kernel_size=init_cfg['kernel_size'], 
                      stride=init_cfg['stride'], 
                      padding=init_cfg['padding'], 
                      bias=False),
            nn.BatchNorm2d(init_cfg['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.maxpool_initial = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # RESIDUAL BLOCKS - DYNAMIC BLOCK CREATION
        blocks_list = []
        for i, block_cfg in enumerate(config_data['residual_blocks_config']):
            in_c = block_cfg['in_channels']
            out_c = block_cfg['out_channels']
            stride = block_cfg['stride']
            
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
            
            block = ResBottleneckBlock2D(
                in_c,
                out_c,
                kernel_size=block_cfg['kernel_size'],
                stride=stride,
                expansion=block_cfg['expansion'],
                dropout=dropout,
                downsample=downsample
            )
            
            blocks_list.append((f'res_block{i+1}', block))

        # Registering dynamically created blocks as modules
        self.residual_blocks = nn.Sequential(OrderedDict(blocks_list))
        
        # The last channel number after all blocks
        final_channels = config_data['residual_blocks_config'][-1]['out_channels']
        
        head_cfg = config_data['head_layers']
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(final_channels, head_cfg['fc1_output_size']),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(head_cfg['fc1_output_size'], num_classes)
        )

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], num_classes: int):
        """
        Load model from a JSON config file
        
        Parameters
        ----------
        config_path : str or Path
            Path to JSON config file
        num_classes : int
            Number of output classes
            
        Returns
        -------
        RandLANet
            Model instance
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(model_config=config, num_classes=num_classes)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.initial_conv(x)
        x = self.maxpool_initial(x)

        # Passing through a dynamically created sequence of blocks
        x = self.residual_blocks(x) 

        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x