import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingBlock(nn.Module):

    def __init__(self, num_channels_in, num_channels_out, filter_size, params):
        """
        Initializes upsampling block. halves number of channels in the sample, doubles dimension

        Args:
            num_channels_in (int): number of channels on input size of upsample
            num_channels_out (int): number of channels on output size of
                upsample convolution
            filter_size (int): number of filters
        """
        super(UpsamplingBlock, self).__init__()

        self.conv = nn.Conv1d(num_channels_in, num_channels_out, filter_size, stride=1, padding = filter_size//2)
        self.drop = nn.Dropout(p=params['drop_prob'])
        self.relu = nn.LeakyReLU(negative_slope=params['relu'])
        self.shuffle = SubpixelShuffle()
        self.proj = nn.Linear(num_channels_out, num_channels_out//2, bias=False)



    def forward(self, input_signal, residual):
        """
        Returns result of upsampling bock

        Args:
            input_signal (torch.tensor): signal to upsample
            residual (torch.tensor): corresponding shape signal from
                downsampling to concatenate with upsample
        """

        # print('in up block\nshape of input:{}\nshape of residual:{}'.format(
        #     input_signal.shape, residual.shape
        # ))

        out = self.conv(input_signal)
        # print('shape after conv:', out.shape)
        out = self.drop(out)
        out = self.relu(out)
        out = self.shuffle(out)
        # print('shape after shuffle:', out.shape)
        proj_res = self.proj(residual.permute(0,2,1)).permute(0,2,1)
        # (b, F/2, d) -> (b, F/4, d)
        out = torch.cat([out, proj_res], dim=1)
        return out


class SubpixelShuffle(nn.Module):
    """ Subpixel shuffling layer to increase the time dimension in upsampling

    1-Dimensional adaptation of Pytorch PixelShuffle module

    """

    def __init__(self, ratio=2):
        super(SubpixelShuffle, self).__init__()
        self.ratio = ratio


    def forward(self, signal):
        """
        Takes as input a signal of size (*, F/2, d) and returns (*, F/4, 2d)

        Args:
            signal (Torch.tensor): size (batch, num_filters, spatial_dim)
        """
        batch_size, in_filters, in_d = signal.shape

        out_filters = in_filters // self.ratio
        out_d = in_d * self.ratio
        in_view = signal.contiguous().view(batch_size, out_filters, self.ratio, in_d)
        # (batch, f/r, r, d)
        shuffle_out = in_view.permute(0,1,3,2).contiguous()  # (b, f/r, d, r)
        return shuffle_out.view(batch_size, out_filters, out_d) # (b, f/r, d*r)
