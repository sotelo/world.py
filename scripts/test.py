if __name__ == "__main__":
    import scipy.io.wavfile
    import matplotlib.pyplot as plt
    import copy
    import numpy as np

    file = 'test16k.wav'
    fs, x = scipy.io.wavfile.read(file)
    from world import *
    fs, nbit, x_length, x = readwav(file)
    #plt.plot(x)
    #plt.show()

    period = 5.0

    opt = pyDioOption(40.0, 700, 2.0, period, 4)

    f0, time_axis = dio(x, fs, period, opt)

    f0_by_dio = copy.deepcopy(f0)
    f0 = stonemask(x, fs, period, time_axis, f0)

    #plt.plot(time_axis, f0)
    #plt.plot(time_axis, f0_by_dio)
    #plt.show()

    spectrogram = star(x, fs, period, time_axis, f0)

    #plt.imshow(np.log(spectrogram))
    #plt.show()

    spectrogram = cheaptrick(x, fs, period, time_axis, f0)

    #plt.imshow(np.log(spectrogram).T,origin="lower", aspect="auto")
    #plt.show()

    #double free
    residual = platinum(x, fs, period, time_axis, f0, spectrogram)

    #plt.imshow(residual.T,origin="lower", aspect="auto")
    #plt.show()
    #print residual.shape
    #print residual
    #print residual.shape

    y = synthesis(fs, period, f0, spectrogram, residual, len(x))
    #print y

    #plt.plot(range(len(y)),y, "r-+")
    #plt.plot(range(len(x)),x)
    #plt.show()

    aperiodicity =  aperiodicityratio(x, fs, period, time_axis, f0)
    #plt.imshow(aperiodicity.T,origin="lower", aspect="auto")
    #plt.show()

    ya = synthesis_from_aperiodicity(fs, period, f0, spectrogram, aperiodicity, len(x))
    #plt.plot(range(len(ya)),ya, "r-+")
    #plt.plot(range(len(x)),x)
    #plt.show()

    writewav(y, fs, nbit, "y_test.wav")
    writewav(ya, fs, nbit, "ya_test.wav")
