"""
Audio Super Res Pytorch Sanity Check
"""

import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from model.layers.downsampling import DownsamplingBlock
from model.layers.upsampling import UpsamplingBlock, SubpixelShuffle
from model.model import AudioUNet, snr, lsd
from model_v2.model import AudioUNetV2

import utils
import model.data_loader as data_loader


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', action='store_true', help='layer sanity check')
    parser.add_argument('--full', action='store_true', help='full sanity check')
    parser.add_argument('--data', action='store_true', help='data load sanity check')
    parser.add_argument('--metric', action='store_true', help='metric sanity check')

    parser.add_argument('--train', default='data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5', help="Train data path")
    parser.add_argument('--val', default='data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5', help="Val data path")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    return args, params

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

def full_sanity_check(params):
    batch_size = 30
    F = 1
    d = 2**13
    in_shape = (batch_size, d, F)
    in_signal = torch.randn(in_shape)
    model = AudioUNetV2(params.blocks, params)

    out = model.forward(in_signal)
    assert(in_signal.shape == out.shape)
    print('passed full sanity check')

def data_check(args, params):
    data_paths = {
        'train' : args.train,
        'val'   : args.val
    }

    dataloaders = data_loader.fetch_dataloader(['train', 'val'], data_paths, params)

def metric_check(args, params):
    data_paths = {
        'train' : args.train,
        'val'   : args.val
    }
    dataloaders = data_loader.fetch_dataloader(['train'], data_paths, params)
    training_set = dataloaders['train']
    for x,y in training_set:
        break
    x, y = np.array(x), np.array(y)
    signal_noise_ratio = snr(x,y)
    log_spec_dist = lsd(x,y)
    print(signal_noise_ratio, log_spec_dist)

def main():
    args, params = parseArgs()
    if args.layers:
        downsampling_sanity_check()
        shuffle_sanity_check()
        upsampling_sanity_check()
    if args.full:
        full_sanity_check(params)
    if args.data:
        data_check(args, params)
    if args.metric:
        metric_check(args, params)

if __name__ == '__main__':
    main()
