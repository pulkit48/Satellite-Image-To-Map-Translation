
import torch
import torch.nn as nn


def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    return None