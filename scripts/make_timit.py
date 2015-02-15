import numpy as np
import sys
from world import *
from librosa_ports import melspec
from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import as_strided


if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: make_timit.py <timit_npy_file> <timit_phones_file>")
    quit()

period = 5.0
fs = 16000
opt = pyDioOption(40.0, 700, 2.0, period, 4)

input_file = sys.argv[1]
label_file = sys.argv[2]
s_n_components = 64
r_n_components = 100

f0s = []
spectrograms = []
residuals = []
ds = np.load(input_file)
ls = np.load(label_file)
for i, X in enumerate(ds[:100]):
    X = X.astype('float64')
    f0, time_axis = dio(X, fs, period, opt)
    f0 = stonemask(X, fs, period, time_axis, f0)
    spectrogram = cheaptrick(X, fs, period, time_axis, f0)
    mel_spectrogram = melspec(spectrogram, fs, s_n_components)
    log_mel_spectrogram = np.log(mel_spectrogram)
    residual = platinum(X, fs, period, time_axis, f0, spectrogram)
    residuals.append(residual)
    spectrograms.append(spectrogram)

compression_residuals = np.concatenate([r[:100] for r in residuals], axis=0)

mel_spectrogram = melspec(spectrograms[0], fs, s_n_components)
log_mel_spectrogram = np.log(mel_spectrogram)
tf_r = PCA(n_components=r_n_components)
tf_r.fit(compression_residuals)

residual_matrix = tf_r.components_
reduced_residual = tf_r.transform(residuals[0])
tf_r_mean = tf_r.mean_

np.save('residual_matrix.npy', residual_matrix)
np.save('residual_mean.npy', tf_r_mean)

data = []
labels = []
for i, X in enumerate(ds):
    print("Processing line %i of %i" % (i, len(ds)))
    X = X.astype('float64')
    f0, time_axis = dio(X, fs, period, opt)
    f0 = stonemask(X, fs, period, time_axis, f0)
    spectrogram = cheaptrick(X, fs, period, time_axis, f0)
    mel_spectrogram = melspec(spectrogram, fs, s_n_components)
    log_mel_spectrogram = np.log(mel_spectrogram)
    residual = platinum(X, fs, period, time_axis, f0, spectrogram)
    reduced_residual = tf_r.transform(residual)
    d = np.concatenate((f0[:, None], log_mel_spectrogram, reduced_residual),
                       axis=1)
    l = ls[i]
    # 40 ms * 16 kHz
    win_size = 40 * 16
    # 5 ms * 16 kHz
    step_size = 5 * 16
    sz = l.itemsize
    shape = (len(d), win_size)
    strides = (sz * step_size, sz)
    # Reshape to 2D array with proper overlap and stride
    reshaped_labels = as_strided(ls[i], shape=shape, strides=strides)
    data.append(d)
    labels.append(reshaped_labels)


data = np.asarray(data)
labels = np.asarray(labels)
np.save('vocoded_TIMIT.npy', data)
np.save('reshaped_TIMIT_labels.npy', labels)
