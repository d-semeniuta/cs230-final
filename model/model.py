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


class AudioUNet(nn.Module):

    def __init__(self, input_depth, num_blocks, params):
        """

        Args:
            input_depth (int): initial number of channels in input signal
            num_blocks (int): number of down and upscaling blocks
            ratio (int): upsampling ratio of the audio
            params (dict): additional parameters of the network
                params['drop_prob'] (float): dropout probability
                params['relu'] (float): leaku relu slope
        """
        super(AudioUNet, self).__init__()
        self.B = num_blocks

        down_num_channels = [min(512, 2**(6+b+1)) for b in range(num_blocks)]
        down_filter_lengths = [max(9, 2**(7-(b+1))+1) for b in range(num_blocks)]
        up_num_channels = list(reversed(down_num_channels))
        up_filter_lengths = list(reversed(down_filter_lengths))
        # up_num_channels = [min(512, 2**(7+num_blocks-b+1)) for b in range(num_blocks)]
        # up_filter_lengths = [max(9, 2**(7-(num_blocks-b+1))+1) for b in range(num_blocks)]

        # down_num_channels = [2**(6+b+1) for b in range(num_blocks)]
        # down_filter_lengths = [2**(7-(b+1))+1 for b in range(num_blocks)]
        # up_num_channels = [2**(7+num_blocks-b+1) for b in range(num_blocks)]
        # up_filter_lengths = [2**(7-(num_blocks-b+1))+1 for b in range(num_blocks)]

        self.down_blocks = [DownsamplingBlock(
            input_depth,
            down_num_channels[0],
            down_filter_lengths[0],
            params
        )]
        self.up_blocks = [UpsamplingBlock(
            down_num_channels[-1],
            up_num_channels[0],
            up_filter_lengths[0],
            params
        )]
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

        # self.params = params


    def forward(self, signal):
        """ Passes the signal through the convolutional bottleneck
        Args:
            signal (torch.tensor): (batch, length, num_channels)
        """
        x = signal
        print('input shape:', signal.shape)
        # downsampling
        residuals = []
        for b in range(self.B):
            x = self.down_blocks[b].forward(x)
            residuals.append(x)
            print('shape after down block {}: {}'.format(b+1, x.shape))


        x = self.bottleneck.forward(x)
        print('shape after bottleneck: {}'.format(x.shape))

        # upsampling
        for b in range(self.B):
            res = residuals[self.B - b - 1]
            x = self.up_blocks[b].forward(x, res)
            print('shape after up block {}: {}'.format(b+1, x.shape))


        x = self.final_conv(x)
        x = self.final_shuffle(x)
        print('shape after final conv: {}'.format(x.shape))

        out = torch.add(x, signal)
        print('shape after adding:', out.shape)
        return out



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
