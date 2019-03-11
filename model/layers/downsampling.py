import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplingBlock(nn.Module):

    def __init__(self, num_channels_in, num_channels_out, filter_size, params):
        """
        Initializes downsampling block. Doubles number of channels in the sample, halves dimension

        Args:
            num_channels_in (int): number of channels on input size of downsample
            num_channels_out (int): number of channels on output size of downsample
            filter_size (int): number of filters
        """
        super(DownsamplingBlock, self).__init__()

        self.conv = nn.Conv1d(num_channels_in, num_channels_out, filter_size, stride=2, padding=filter_size//2)
        self.norm = nn.BatchNorm1d(num_channels_out)
        self.relu = nn.LeakyReLU(params.relu_slope)


    def forward(self, input_signal):
        """

        """
        out = self.conv(input_signal)
        out = self.norm(out)
        out = self.relu(out)
        return out
