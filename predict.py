"""
Used for upscaling audio files
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from model.model import AudioUNet
import model.model as model_params
import utils
from model.audio_process import upsample_wav

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True,
            help="Directory containing model checkpoints")
    parser.add_argument('--file_list', required=True,
            help="List of files to run the model on")
    parser.add_argument('--file_dir', required=True,
            help="Directory of files to run the model on")
    parser.add_argument('--r', required=True, type=int,
            help="upscaling factor")
    parser.add_argument('--sr', default=16000, type=int,
            help="hi-res sampling rate")
    args = parser.parse_args()
    # load model params
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    return args, params

def runPredictions(args, params):
    model = AudioUNet(params.blocks, params)
    utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
    file_dir = args.file_dir
    out_dir = args.model_dir + '/predictions/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.file_list) as f:
        for line in f:
            try:
                print('Upsampling {}'.format(line))
                file = os.path.join(file_dir, line.strip())
                upsample_wav(file, args, model, out_dir)
            except EOFError:
                print('Error reading {}'.format(line))

def main():
    args, params = parseArgs()
    runPredictions(args, params)

if __name__ == '__main__':
    main()
