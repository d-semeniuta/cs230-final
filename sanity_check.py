"""
Audio Super Res Pytorch Sanity Check
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from model.layers.downsampling import DownsamplingBlock
from model.layers.upsampling import UpsamplingBlock, SubpixelShuffle

def downsampling_sanity_check():
    batch_size = 20
    in_filters = 2**6
    in_dim = 1024
    in_shape = (batch_size, in_filters, in_dim)
    in_signal = torch.randn(in_shape)

    num_channels_in = 2**6
    num_channels_out = num_channels_in*2
    filter_size = 9
    ds = DownsamplingBlock(num_channels_in, num_channels_out, filter_size)

    out_signal = ds.forward(in_signal)
    assert(out_signal.shape == (batch_size, in_filters*2, in_dim/2))
    print('passed downsampling sanity check')

def shuffle_sanity_check():
    in_shape = (20, 2**6, 1024)
    r = 2
    in_signal = torch.randn(in_shape)
    sps = SubpixelShuffle(r)
    out = sps.forward(in_signal)
    assert(out.shape == (in_shape[0], in_shape[1]/r, in_shape[2]*r))
    print('passed shuffle sanity check')

def upsampling_sanity_check():
    batch_size = 20
    F = 2**6
    d = 1024
    in_shape = (batch_size, F, d)
    in_signal = torch.randn(in_shape)

    res_shape = (batch_size, F//4, 2*d)
    res_signal = torch.randn(res_shape)

    num_channels_in = 2**6
    num_channels_out = num_channels_in//2
    filter_size = 9
    us = UpsamplingBlock(num_channels_in, num_channels_out, filter_size, 2)
    out_signal = us.forward(in_signal, res_signal)

    assert(out_signal.shape == (batch_size, F//2, 2*d))
    print('passed upsampling sanity check')

def main():
    downsampling_sanity_check()
    shuffle_sanity_check()
    upsampling_sanity_check()


if __name__ == '__main__':
    main()
