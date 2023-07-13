import torch
import torch.nn as nn
from lib.nn import SynchronizedBatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    """
    Returns a 3x3 convolution with padding to preserve spatial dimensions.
    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): convolution stride, default is 1
        has_bias (bool): whether to include a bias term, default is False
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """
    Returns a 3x3 convolution with padding, followed by batch normalization and a ReLU activation.
    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): convolution stride, default is 1
    """
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )