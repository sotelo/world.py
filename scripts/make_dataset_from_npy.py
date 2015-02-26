import glob
import numpy as np
import os
import sys
import tables
from world import *
from librosa_ports import melspec
from sklearn.decomposition import PCA


if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: make_dataset.py <directory_of_npy_files> [save_path=<directory_of_npy_files>/saved_world_data.h5]")
    quit()

files_dir = sys.argv[1]
npy_files = glob.glob(os.path.join(files_dir, '*[_,0-9][0-9].npy'))
npy_files = sorted(npy_files, key=lambda x: int(x.split("_")[-1][:-4]))
# Only do 500k examples at first, 10k per file
npy_files = npy_files[101:]

if len(sys.argv) > 2:
    h5_file_path = os.path.join(sys.argv[2])
else:
    h5_file_path = os.path.join(files_dir, "saved_world_data.h5")

if os.path.exists(h5_file_path):
    raise ValueError(
        "File already exists at %s, exiting.\n \
         Delete and rerun script if you want a new dataset!!!" % h5_file_path)

period = 5.0
fs = 16000
opt = pyDioOption(40.0, 700, 2.0, period, 4)

# Get relevant shapes for EArray and create principal component matrices
log_spectrograms = []
residuals = []
ds = np.load(npy_files[0])
for i, X in enumerate(ds[:100]):
    print("Fetching line %i of %i" % (i, len(ds)))
    X = X.astype('float64')
    f0, time_axis = dio(X, fs, period, opt)
    f0 = stonemask(X, fs, period, time_axis, f0)
    spectrogram = cheaptrick(X, fs, period, time_axis, f0)
    residual = platinum(X, fs, period, time_axis, f0, spectrogram)
    log_spectrogram = np.log(spectrogram)
    residuals.append(residual)
    log_spectrograms.append(spectrogram)

log_spectrograms = np.concatenate(log_spectrograms, axis=0)
residuals = np.concatenate(residuals, axis=0)

s_n_components = 64
r_n_components = 100

print("Calculating compressed spectrogram exmaple")
mel_spectrogram = melspec(spectrogram, fs, s_n_components)
log_mel_spectrogram = np.log(mel_spectrogram)
print("Calculating decomposition of residual subset")
tf_r = PCA(n_components=r_n_components)
tf_r.fit(residuals)

residual_matrix = tf_r.components_
reduced_residual = tf_r.transform(residual)
tf_r_mean = tf_r.mean_

h5_file = tables.openFile(h5_file_path, mode='w')
print("Creating dataset at %s from directory %s" % (files_dir, h5_file_path))
compression_filter = tables.Filters(complevel=5, complib='blosc')
residual_matrix_storage = h5_file.createCArray(h5_file.root, 'residual_matrix',
                                               tables.Float32Atom(),
                                               shape=residual_matrix.shape,
                                               filters=compression_filter)
residual_subset_mean = h5_file.createCArray(h5_file.root,
                                            'residual_subset_mean',
                                            tables.Float32Atom(),
                                            shape=tf_r_mean.shape,
                                            filters=compression_filter)

residual_matrix_storage[:] = residual_matrix.astype('float32')
residual_subset_mean[:] = tf_r_mean.astype('float32')

feature_sizes = h5_file.createEArray(h5_file.root, 'feature_sizes',
                                     tables.Float32Atom(),
                                     shape=(0, 2),
                                     filters=compression_filter)

feature_sizes.append(np.asarray(f0[:, None].shape)[None])
feature_sizes.append(np.asarray(log_mel_spectrogram.shape)[None])
feature_sizes.append(np.asarray(reduced_residual.shape)[None])

total_features = f0[:, None].shape[1] + log_mel_spectrogram.shape[1] + reduced_residual.shape[1]

data = h5_file.createEArray(h5_file.root, 'data',
                            tables.Float32Atom(),
                            shape=(0, log_mel_spectrogram.shape[0], total_features),
                            filters=compression_filter)
meta_info = h5_file.createEArray(h5_file.root, 'meta_info',
                                 tables.StringAtom(itemsize=8),
                                 shape=(0,),
                                 filters=compression_filter)
original_length = h5_file.createEArray(h5_file.root, 'original_length',
                                       tables.Int32Atom(),
                                       shape=(0, 1),
                                       filters=compression_filter)
for n, npy_file in enumerate(npy_files):
    ds = np.load(npy_file)
    print("Processing file %i of %i: %s" % (n, len(npy_files), npy_file))
    for i, X in enumerate(ds):
        print("Processing line %i of %i" % (i, len(ds)))
        X = X.astype('float64')
        f0, time_axis = dio(X, fs, period, opt)
        f0 = stonemask(X, fs, period, time_axis, f0)
        spectrogram = cheaptrick(X, fs, period, time_axis, f0)
        residual = platinum(X, fs, period, time_axis, f0, spectrogram)
        log_mel_spectrogram = np.log(melspec(spectrogram, fs, 64))
        residual = tf_r.transform(residual)
        sample = np.concatenate((f0[:, None], log_mel_spectrogram, residual),
                                axis=1)
        data.append(sample.astype('float32')[None])
        meta_info.append(np.array(["%s-%i" % (npy_file, i)], dtype='S8'))
        original_length.append(np.asarray([len(X), ]).astype('int32')[None])
h5_file.close()
