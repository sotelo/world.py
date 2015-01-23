import numpy as np


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    :usage:
        >>> librosa.mel_to_hz(3)
        array([ 200.])

        >>> librosa.mel_to_hz([1,2,3,4,5])
        array([  66.66666667,  133.33333333,  200.        ,  266.66666667,
                333.33333333])

    :parameters:
      - mels          : np.ndarray [shape=(n,)], float
          mel bins to convert
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - frequencies   : np.ndarray [shape=(n,)]
          input mels in Hz
    """

    mels = np.asarray([mels], dtype=float).flatten()

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :usage:
        >>> librosa.hz_to_mel(60)
        array([0.9])
        >>> librosa.hz_to_mel([110, 220, 440])
        array([ 1.65,  3.3 ,  6.6 ])

    :parameters:
      - frequencies   : np.ndarray [shape=(n,)] , float
          scalar or array of frequencies
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - mels        : np.ndarray [shape=(n,)]
          input frequencies in Mels
    """

    frequencies = np.asarray([frequencies]).flatten()

    if np.isscalar(frequencies):
        frequencies = np.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False,
                    extra=False):
    """Compute the center frequencies of mel bands

    :usage:
        >>> librosa.mel_frequencies(n_mels=40)
        array([    0.        ,    81.15543818,   162.31087636,   243.46631454,
                324.62175272,   405.7771909 ,   486.93262907,   568.08806725,
                649.24350543,   730.39894361,   811.55438179,   892.70981997,
                973.86525815,  1058.38224675,  1150.77458676,  1251.23239132,
                1360.45974173,  1479.22218262,  1608.3520875 ,  1748.75449257,
                1901.4134399 ,  2067.39887435,  2247.87414245,  2444.10414603,
                2657.46420754,  2889.44970936,  3141.68657445,  3415.94266206,
                3714.14015814,  4038.36904745,  4390.90176166,  4774.2091062 ,
                5190.97757748,  5644.12819182,  6136.83695801,  6672.55713712,
                7255.04344548,  7888.37837041,  8577.0007833 ,  9325.73705043])

    :parameters:
      - n_mels    : int > 0 [scalar]
          number of Mel bins

      - fmin      : float >= 0 [scalar]
          minimum frequency (Hz)

      - fmax      : float >= 0 [scalar]
          maximum frequency (Hz)

      - htk       : bool
          use HTK formula instead of Slaney

      - extra     : bool
          include extra frequencies necessary for building Mel filters

    :returns:
      - bin_frequencies : ndarray [shape=(n_mels,)]
          vector of Mel frequencies
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz_to_mel(fmin, htk=htk)
    maxmel = hz_to_mel(fmax, htk=htk)

    mels = np.arange(minmel, maxmel + 1, (maxmel - minmel) / (n_mels + 1.0))

    if not extra:
        mels = mels[:n_mels]

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of ``np.fft.fftfreqs``

    :usage:
        >>> librosa.fft_frequencies(sr=22050, n_fft=16)
        array([     0.   ,   1378.125,   2756.25 ,   4134.375,   5512.5  ,
                 6890.625,   8268.75 ,   9646.875,  11025.   ])

    :parameters:
      - sr : int > 0 [scalar]
          Audio sampling rate

      - n_fft : int > 0 [scalar]
          FFT window size

    :returns:
      - freqs : np.ndarray [shape=(1 + n_fft/2,)]
          Frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft/2),
                       endpoint=True)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :usage:
        >>> mel_fb = librosa.filters.mel(22050, 2048)

        >>> # Or clip the maximum frequency to 8KHz
        >>> mel_fb = librosa.filters.mel(22050, 2048, fmax=8000)

    :parameters:
      - sr        : int > 0 [scalar]
          sampling rate of the incoming signal

      - n_fft     : int > 0 [scalar]
          number of FFT components

      - n_mels    : int > 0 [scalar]
          number of Mel bands to generate

      - fmin      : float >= 0 [scalar]
          lowest frequency (in Hz)

      - fmax      : float >= 0 [scalar]
          highest frequency (in Hz).
          If ``None``, use ``fmax = sr / 2.0``

      - htk       : bool [scalar]
          use HTK formula instead of Slaney

    :returns:
      - M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
          Mel transform matrix
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft / 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = mel_frequencies(n_mels,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk,
                            extra=True)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - fftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


def melspec(S, fs, nmelbands=128):
    nfft = 2 * (S.shape[1] - 1)
    melfb = mel(fs, nfft)
    return melfb.dot(S.T).T


def invmelspec(M, fs, nfft=1024):
    melfb = mel(fs, nfft, n_mels=M.shape[1])
    ww = np.dot(melfb.T, melfb)
    iwts = (melfb / np.maximum(np.mean(np.diag(ww)) / 100,
                               np.sum(ww, axis=1))).T
    return np.dot(iwts, M.T).T
