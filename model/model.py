"""
Pytorch implementation of Audio Super-Resolution

Author:
    Daniel Semeniuta (dsemeniu@cs.stanford.edu)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.downsampling import DownsamplingBlock
from model.layers.upsampling import UpsamplingBlock, SubpixelShuffle
import model.audio_process as ap

class AudioUNet(nn.Module):

    def __init__(self, num_blocks, params, input_depth=1):
        """

        Args:
            input_depth (int): initial number of channels in input signal
            num_blocks (int): number of down and upscaling blocks
            params (dict): additional parameters of the network
                params['drop_prob'] (float): dropout probability
                params['relu_slope'] (float): leaky relu slope
                params['max_channels'] (int): max number of channels in convolution
                params['min_filter'] (int): min size of filter in conv
        """
        super(AudioUNet, self).__init__()
        self.B = num_blocks
        max_channels = params.max_channels
        min_filter = params.min_filter

        down_num_channels = [min(max_channels, 2**(6+b))
            for b in range(1, num_blocks+1)]
        down_filter_lengths = [max(min_filter, 2**(7-b) + 1)
            for b in range(1, num_blocks+1)]
        up_num_channels = list(reversed(down_num_channels))
        up_filter_lengths = list(reversed(down_filter_lengths))

        self.down_blocks = nn.ModuleList([DownsamplingBlock(
            input_depth,
            down_num_channels[0],
            down_filter_lengths[0],
            params
        )])
        self.up_blocks = nn.ModuleList([UpsamplingBlock(
            down_num_channels[-1],
            up_num_channels[0],
            up_filter_lengths[0],
            params
        )])
        for b in range(1,num_blocks):
            self.down_blocks.append(DownsamplingBlock(
                down_num_channels[b-1],
                down_num_channels[b],
                down_filter_lengths[b],
                params
            ))
            self.up_blocks.append(UpsamplingBlock(
                up_num_channels[b-1],
                up_num_channels[b],
                up_filter_lengths[b],
                params
            ))

        self.bottleneck = DownsamplingBlock(
            down_num_channels[-1],
            down_num_channels[-1],
            down_filter_lengths[-1],
            params
        )

        self.final_conv = nn.Conv1d(
            up_num_channels[-1],
            2*input_depth,
            9,
            padding = 9//2
        )
        self.final_shuffle = SubpixelShuffle()

    def forward(self, signal):
        """ Passes the signal through the convolutional bottleneck
        Args:
            signal (torch.tensor): (batch, length, num_channels)
        """
        # (batch, len, num_channels) -> (batch, num_channels, len)
        x = signal.permute(0,2,1)

        # downsampling
        residuals = []
        for b in range(self.B):
            x = self.down_blocks[b].forward(x)
            residuals.append(x)

        x = self.bottleneck.forward(x)

        # upsampling
        for b in range(self.B):
            res = residuals[self.B - b - 1]
            x = self.up_blocks[b].forward(x, res)

        x = self.final_conv(x)
        x = self.final_shuffle.forward(x).permute(0,2,1)

        out = torch.add(x, signal)
        return out

loss_fn = nn.MSELoss()

def snr(x, y):
    """
    Compute the signal to noise ratio, given the approximation and reference
        signals
    Args:
        x: approximation
        y: reference

    Returns: (float) ratio
    """
    x, y = np.squeeze(x, axis=-1), np.squeeze(y, axis=-1)
    return np.mean(10 * np.log(np.linalg.norm(y, axis=-1)**2 / np.linalg.norm(x - y, axis=-1)**2))

def lsd(x, y):
    """ Computes Log-spectral distance
    adapted from jhetherly/EnglishSpeechUpsampler
    """
    X, Y = np.squeeze(x, axis=-1), np.squeeze(y, axis=-1)
    lsds = []
    for x,y in zip(X,Y):
        x_gram = ap.get_spectogram(x)
        y_gram = ap.get_spectogram(y)
        X = np.log10(np.abs(x_gram)**2)
        Y = np.log10(np.abs(y_gram)**2)
        diff_squared = (X - Y)**2
        lsd = np.mean(np.sqrt(np.mean(diff_squared, axis=0)))
        lsds.append(lsd)
    return np.mean(np.array(lsds))

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'snr': snr,
    'lsd': lsd
}
