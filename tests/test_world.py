import os
import copy
import numpy as np
from world import *

import unittest


class WorldTestCase(unittest.TestCase):

    def test_synthesis(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', 'scripts', 'test16k.wav')
        fs, nbit, x_length, x = readwav(path)
        x = np.array(x, dtype=np.float)
        period = 5.0
        opt = pyDioOption(40.0, 700, 2.0, period, 4)

        f0, time_axis = dio(x, fs, period, opt)
        assert isinstance(f0, np.ndarray)
        assert isinstance(time_axis, np.ndarray)
        assert not any(np.isnan(f0))

        f0 = stonemask(x, fs, period, time_axis, f0)
        spectrogram = cheaptrick(x, fs, period, time_axis, f0)
        assert isinstance(spectrogram, np.ndarray)
        assert not np.isnan(spectrogram).all()

        residual = platinum(x, fs, period, time_axis, f0, spectrogram)
        assert isinstance(residual, np.ndarray)
        assert not np.isnan(residual).all()

        y = synthesis(fs, period, f0, spectrogram, residual, len(x))
        assert isinstance(residual, np.ndarray)
        assert not np.isnan(y).all()

    def test_ap_synthesis(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', 'scripts', 'test16k.wav')
        fs, nbit, x_length, x = readwav(path)
        period = 5.0
        opt = pyDioOption(40.0, 700, 2.0, period, 4)

        f0, time_axis = dio(x, fs, period, opt)
        assert isinstance(f0, np.ndarray)
        assert isinstance(time_axis, np.ndarray)
        assert not any(np.isnan(f0))

        f0 = stonemask(x, fs, period, time_axis, f0)
        spectrogram = cheaptrick(x, fs, period, time_axis, f0)
        assert isinstance(spectrogram, np.ndarray)
        assert not np.isnan(spectrogram).all()

        aperiodicity = aperiodicityratio(x, fs, period, time_axis, f0)
        assert isinstance(aperiodicity, np.ndarray)
        assert not np.isnan(aperiodicity).all()

        ya = synthesis_from_aperiodicity(
            fs, period, f0, spectrogram, aperiodicity, len(x))
        assert isinstance(ya, np.ndarray)
        assert not np.isnan(ya).all()
