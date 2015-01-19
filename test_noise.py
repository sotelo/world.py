from scipy.io import wavfile
import copy
import numpy as np
import os
from wrap1 import *

def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.
    Parameters
    ----------
    X : ndarray
        Signal to be rescaled
    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.
    Returns
    -------
    X_sc : ndarray
        (-1, 1) scaled version of X as float, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X *= 2 ** 15
    return X.astype('int16')

def test_noise(noise_coeff=0.00):
    file = 'test16k.wav'
    fs, x = wavfile.read(file)
    fs, nbit, x_length, x = readwav(file)
    period = 5.0
    opt = pyDioOption(40.0, 700, 2.0, period, 4)

    f0, time_axis = dio(x, fs, period, opt)

    f0_by_dio = copy.deepcopy(f0)
    f0 = stonemask(x, fs, period, time_axis, f0)
    spectrogram = star(x, fs, period, time_axis, f0)
    spectrogram = cheaptrick(x, fs, period, time_axis, f0)
    residual = platinum(x, fs, period, time_axis, f0, spectrogram)
    random_state = np.random.RandomState(1999)
    spectrogram += noise_coeff * random_state.randn(*spectrogram.shape)
    residual += noise_coeff * random_state.randn(*residual.shape)
    y = synthesis(fs, period, f0, spectrogram, residual, len(x))
    aperiodicity = aperiodicityratio(x, fs, period, time_axis, f0)
    aperiodicity += noise_coeff * random_state.randn(*aperiodicity.shape)
    ya = synthesis_from_aperiodicity(fs, period, f0, spectrogram, aperiodicity, len(x))
    noise_str = str(noise_coeff).split('.')[-1]
    wavfile.write("y_%s.wav" % noise_str, fs, soundsc(y))
    wavfile.write("ya_%s.wav" % noise_str, fs, soundsc(ya))

if __name__ == "__main__":
    test_noise(1e-16)
    test_noise(1e-14)
    test_noise(1e-12)
    test_noise(1e-10)
    test_noise(1e-8)
    test_noise(1e-6)
    test_noise(1e-4)
    test_noise()
