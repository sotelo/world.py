from scipy.io import wavfile
import glob
import copy
import numpy as np
import os
import sys
from world import *
import matplotlib.pyplot as plt
#from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: make_dataset.py <directory_of_wav_files>")
    quit()

files_dir = sys.argv[1]
wav_files = glob.glob(os.path.join(files_dir, "*.wav"))
spectrogram_data = []
residual_data = []
f0_data = []
file_keys = []
period = 5.0
opt = pyDioOption(40.0, 700, 2.0, period, 4)
for n, wav_file in enumerate(wav_files):
    print("Processing file %i : %s" % (n, wav_file))
    fs, X = wavfile.read(wav_file)
    X = X.astype('float64')
    f0, time_axis = dio(X, fs, period, opt)
    f0 = stonemask(X, fs, period, time_axis, f0)
    spectrogram = cheaptrick(X, fs, period, time_axis, f0)
    residual = platinum(X, fs, period, time_axis, f0, spectrogram)
    residual[residual < 1e-6] = 1e-6
    log_spectrogram = np.log(spectrogram)
    f0_data.append(f0)
    spectrogram_data.append(log_spectrogram)
    residual_data.append(residual)
    file_keys.append(wav_file)

def normalize(X):
    fixed_std = X.std(axis=0)
    return ((X - X.mean(axis=0)) / fixed_std,
            X.mean(axis=0), fixed_std)

log_spectrograms = np.concatenate(spectrogram_data, axis=0)
residuals = np.concatenate(residual_data, axis=0)

tf_s = PCA(n_components=100)
tf_s.fit(log_spectrograms)
tf_r = PCA(n_components=100)
tf_r.fit(residuals)

mean_s = np.mean(log_spectrograms, axis=0)
std_s = np.std(log_spectrograms, axis=0)
mean_r = np.mean(residuals, axis=0)
std_r = np.std(residuals, axis=0)

for n, (s, r) in enumerate(zip(spectrogram_data, residual_data)):
    spectrogram_data[n] = tf_s.transform(s)
    residual_data[n] = tf_r.transform(r)

f0s = np.concatenate(f0_data, axis=0)
if len(f0s.shape) < 2:
    f0s = f0s[:, None]
t1 = np.concatenate(spectrogram_data, axis=0)
t2 = np.concatenate(residual_data, axis=0)
t = np.concatenate((f0s, t1, t2), axis=1)

stats_s = np.concatenate((mean_s[:, None], std_s[:, None]), axis=1)
stats_r = np.concatenate((mean_r[:, None], std_r[:, None]), axis=1)
np.save("f0_and_factor_data.npy", t)
np.save("spectrogram_statistics.npy", stats_s)
np.save("residual_statistics.npy", stats_r)
np.save("spectrogram_transform_matrix.npy", tf_s.components_)
np.save("residual_transform_matrix.npy", tf_r.components_)
