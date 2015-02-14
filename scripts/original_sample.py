import numpy as np
import sys
from world import *
from scipy.io import wavfile

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
    print("Usage: make_dataset.py <npy_file>")
    quit()

npy_file = sys.argv[1]
idx = 474
fs = 16000
ds = np.load(npy_file)
y = ds[idx]
wavfile.write("y_o.wav", fs, soundsc(y))
