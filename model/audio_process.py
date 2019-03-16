"""
Utils for processing audio
"""

import numpy as np
import librosa

from scipy.signal import decimate
from scipy import interpolate
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
    """ Adapted from kuleshov/audio-super-res
    Args:
        wav (string) : location of wav file to upsample
        args (args dict) : parameters of run
        model (nn.Module) : pytorch module
        out_dir (string) : where to write the predictions and results
    """
    # load signal
    x_hr, fs = librosa.load(wav, sr=args.sr) # high-res

    # downscale signal
    x_lr = decimate(x_hr, args.r) # low-res
    # x_lr_tensor = torch.from_numpy(x_lr.copy()).double()
    # x_lr_tensor = torch.unsqueeze(torch.unsqueeze(x_lr_tensor, 0), -1)
    # upscale the low-res version
    # out = model_predict(x_lr, model, args.r)

    # get low-res ready for model
    x_sp = spline_up(x_lr, args.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(model.B+1)))]
    X = torch.from_numpy(x_sp.reshape((1,len(x_sp),1)).copy())
    out = model.forward(X)
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

def spline_up(x_lr, r):
    """ adapted from kuleshov/audio-super-res
    Performs a cubic spline on an input so that it can be fed into the model

    Args:
        x_lr (np.array) : low res x input
        r (int) : upscaling ratio
    """
    x_lr = x_lr.squeeze()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp
