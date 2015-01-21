from scipy.io import wavfile
import glob
import copy
import numpy as np
import os
import sys
from wrap1 import *
import matplotlib.pyplot as plt

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

# This should change is we change settings in make_dataset
# Need to get last term from data or fix value somehow
period = 5.0
fs = 16000
len_x = 60700

data = np.load("f0_and_factor_data.npy")
stats_s = np.load("spectrogram_statistics.npy")
stats_r = np.load("residual_statistics.npy")
tf_s = np.load("spectrogram_transform_matrix.npy")
tf_r = np.load("residual_transform_matrix.npy")

f0s = data[:, 0]
log_spectrograms = data[:, 1:1 + tf_s.shape[0]]
residuals = data[:, 1 + tf_s.shape[0]:]
mean_s = stats_s[:, 0]
std_s = stats_s[:, 1]
mean_r = stats_r[:, 0]
std_r = stats_r[:, 1]

log_spectrograms = np.dot(log_spectrograms, tf_s)
residuals = np.dot(residuals, tf_r)

spectrograms = np.exp(log_spectrograms)

s = np.ascontiguousarray(spectrograms)
r = np.ascontiguousarray(residuals)
f0 = np.ascontiguousarray(f0s)
y = synthesis(fs, period, f0, s, r, len_x)
wavfile.write("y_r.wav", fs, soundsc(y))
