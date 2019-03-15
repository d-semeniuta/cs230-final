"""
Utils for processing audio
"""

import numpy as np
import librosa

from scipy.signal import decimate
from matplotlib import pyplot as plt

import torch

def get_spectogram(x, n_fft=2048):
    """ Adapted from kuleshov/audio-super-res """
    S = librosa.stft(x, n_fft)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S

def save_spectogram(S, outfile='spectrogram.png'):
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    plt.savefig(outfile)

def upsample_wav(wav, args, model, out_dir):
    """ Adapted from kuleshov/audio-super-res """
    # load signal
    x_hr, fs = librosa.load(wav, sr=args.sr) # high-res

    # downscale signal
    x_lr = torch.tensor(decimate(x_hr, args.r)) # low-res

    # upscale the low-res version
    out = model.forward(x_lr.reshape((1,len(x_lr),1)))
    x_pr = np.array(out.squeeze()) # pred

    # crop so that it works with scaling ratio
    x_hr = x_hr[:len(x_pr)]
    x_lr = x_lr[:len(x_pr)]

    # save the file
    outname = out_dir + wav + '.out'
    librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)
    librosa.output.write_wav(outname + '.lr.wav', x_lr, fs / args.r)
    librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)

    # save the spectrum
    S = get_spectogram(x_pr, n_fft=2048)
    save_spectogram(S, outfile=outname + '.pr.png')
    S = get_spectogram(x_hr, n_fft=2048)
    save_spectogram(S, outfile=outname + '.hr.png')
    S = get_spectogram(x_lr, n_fft=2048/args.r)
    save_spectogram(S, outfile=outname + '.lr.png')
