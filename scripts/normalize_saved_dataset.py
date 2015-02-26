import glob
import numpy as np
import os
import sys
import tables
from world import *

if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: normalize_saved_dataset.py <hdf5_file> [saved_normalization_file])")
    quit()

h5_file_path = sys.argv[1]

if not os.path.exists(h5_file_path):
    raise ValueError(
        "File doesn't exist at %s, exiting." % h5_file_path)
storage_path = os.path.join(*(os.path.split(h5_file_path)[:-1]))
h5_file = tables.openFile(h5_file_path, mode='a')
data = h5_file.root.data
X = data[0:10000]
orig_shape = X.shape
X = X.reshape(orig_shape[0] * orig_shape[1], -1)
if sys.argv < 3:
    # Check for unreasonably small divisors from constant features and correct to
    # avoid NaN
    min_stats = np.min(X, axis=0)
    max_stats = np.max(X, axis=0)
    max_stats[max_stats < 1.] = 1.
    mean_stats = np.mean(X, axis=0)
    std_stats = np.std(X, axis=0)
    std_stats[std_stats < 1E-6] = 1E-6

    np_path = os.path.join(storage_path, 'min_max_mean_std.npy')
    np.save(np_path,
            np.vstack((min_stats, max_stats, mean_stats, std_stats)))
else:
    s = np.load(sys.argv[2])
    min_stats = s[0]
    max_stats = s[1]
    mean_stats = s[2]
    std_stats = s[3]

data = h5_file.root.data
n_spec = 2 * 64
n_res = 100
# Loop through applying Junyoung's preprocessing
for n, X in enumerate(data):
    print("Processing row %i" % n)
    f0_slice = data[n, :, 0]
    spec_slice = data[n, :, 1:n_spec + 1]
    spec_mean = mean_stats[1:n_spec + 1]
    spec_std = std_stats[1:n_spec + 1]
    res_slice = data[n, :, -n_res:]
    res_mean = mean_stats[-n_res:]
    res_std = std_stats[-n_res:]
    """
    # Test
    print(f0_slice / max_stats[0])
    print((spec_slice - spec_mean) / spec_std)
    print((res_slice - res_mean) / res_std)
    """
    data[n, :, 0] = f0_slice / max_stats[0]
    data[n, :, 1:n_spec + 1] = (spec_slice - spec_mean) / spec_std
    data[n, :, -n_res:] = (res_slice - res_mean) / res_std
