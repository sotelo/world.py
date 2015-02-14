from scipy.io import wavfile
import numpy as np
import os
import sys
import tables
import copy
from world import *
from librosa_ports import invmelspec

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

if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: reconstruct.py <hdf5_file> <stats_file>")

h5_file_path = os.path.join(sys.argv[1])
if not os.path.exists(h5_file_path):
    raise ValueError("File doesn't exist at %s, exiting." % h5_file_path)

m = np.load(sys.argv[3])
min_stats = m[0]
max_stats = m[1]
mean_stats = m[2]
std_stats = m[3]

# This should change is we change settings in make_dataset
# Need to get last term from data or fix value somehow
period = 5.0
fs = 16000

h5_file = tables.openFile(h5_file_path, mode='r')
residual_matrix = h5_file.root.residual_matrix
residual_subset_mean = h5_file.root.residual_subset_mean
meta_info = h5_file.root.meta_info
original_length = h5_file.root.original_length
feature_sizes = h5_file.root.feature_sizes
f0_shape = feature_sizes[0]
log_mel_spectrogram_shape = feature_sizes[1]
reduced_residual_shape = feature_sizes[2]

idx = 474
n_log_mel_components = 2 * 64
n_residual_components = 100

f0_max = max_stats[0]
spec_mean = mean_stats[1:n_log_mel_components + 1]

X = h5_file.root.data[idx]
f0 = X[:, 0] * max_stats[0]
log_mel_spectrogram = X[:, 1:n_log_mel_components + 1] * std_stats[1:128] +
residual = X[:, -n_residual_components:]
mel_spectrogram = np.exp(log_mel_spectrogram)
# 1024 is constant
# Add small constant to avoid NaN
# Needs to be contiguous for WORLD
spectrogram = np.ascontiguousarray(invmelspec(mel_spectrogram, fs, 1024)) + 1E-12

r_means = residual_subset_mean[:]
residual = np.dot(residual, residual_matrix) + r_means

len_x = original_length[idx, 0]

s = np.ascontiguousarray(spectrogram.astype('float64'))
r = np.ascontiguousarray(residual.astype('float64'))
f0 = np.ascontiguousarray(f0.astype('float64'))
period = np.cast['float64'](period)
fs = np.cast['int32'](fs)
len_x = np.cast['int32'](len_x)
s = copy.deepcopy(s)
r = copy.deepcopy(r)
f0 = copy.deepcopy(f0)

y = synthesis(fs, period, f0, s, r, len_x)
wavfile.write("y_r.wav", fs, soundsc(y))
