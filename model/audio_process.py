"""
Utils for processing audio
"""

import numpy as np
import librosa

def get_spectogram(x, n_fft=2048):
    """ Adapted from kuleshov/audio-super-res """
    S = librosa.stft(x, n_fft)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S
