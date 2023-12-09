import numpy as np
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    
    def __init__(self, config_dict):
        super().__init__()
        self.img_size = config_dict['img_size']
        self.img_channels = config_dict['img_channels']
        self.patch_size = config_dict['patch_size']
        self.projection_dim = config_dict['projection_dim']

        self.projection = nn.Conv2d(
            in_channels = self.img_channels,
            out_channels = self.projection_dim,
            stride = self.patch_size,
            kernel_size = self.patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x