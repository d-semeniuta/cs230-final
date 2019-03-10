"""
Pytorch implementation of Audio Super-Resolution

Author:
    Daniel Semeniuta (dsemeniu@cs.stanford.edu)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.downsampling import DownsamplingBlock
from layers.upsampling import UpsamplingBlock


class AudioUNet(nn.Module):

    def __init__(self, num_blocks, ratio):
        """

        Args:
            num_blocks (int): number of down and upscaling blocks
            ratio (int): upsampling ratio of the audio
        """
        super(AudioUNet, self).__init__()



    def forward(self, s):


def snr(x, y):
    """
    Compute the signal to noise ratio, given the approximation and reference
        signals

    Args:
        x: approximation
        y: reference

    Returns: (float) ratio
    """

    return 10 * np.log(np.linalg.norm(y)**2 / np.linalg.norm(x - y)**2)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'snr': snr,
}
