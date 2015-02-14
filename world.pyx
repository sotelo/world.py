import cython
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

import numpy as np
cimport numpy as np

cimport world

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_DOUBLE, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        free(<void*>self.data_ptr)

def readwav(filename):
    cdef np.ndarray[int, ndim=1, mode="c"] fs
    cdef np.ndarray[int, ndim=1, mode="c"] nbit
    cdef np.ndarray[int, ndim=1, mode="c"] x_length
    fs = np.zeros(1, dtype = np.dtype('int32'))
    nbit = np.zeros(1, dtype = np.dtype('int32'))
    x_length = np.zeros(1, dtype = np.dtype('int32'))
    cdef double *array
    cdef np.ndarray ndarray
    array = wavread(filename, &fs[0], &nbit[0], &x_length[0])
    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(x_length[0], <void*> array)
    ndarray = np.array(array_wrapper, copy=False)
    ndarray.base = <PyObject*> array_wrapper
    Py_INCREF(array_wrapper)
    return fs[0], nbit[0], x_length[0], ndarray


def writewav(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, int nbit, filename):
    cdef x_length = len(x)
    wavwrite(&x[0], x_length, fs, nbit, filename)

def dio(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period, option):
    cdef np.ndarray[double, ndim=1, mode="c"] f0
    cdef np.ndarray[double, ndim=1, mode="c"] time_axis
    x_length = len(x)
    f0_length = GetSamplesForDIO(fs, x_length, period)
    f0 = np.zeros(f0_length, dtype = np.dtype('float64'))
    time_axis = np.zeros(f0_length, dtype = np.dtype('float64'))
    Dio(&x[0], x_length, fs, option.option, &time_axis[0], &f0[0])
    return f0, time_axis

def stonemask(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period,
              np.ndarray[double, ndim=1, mode="c"] time_axis not None,
              np.ndarray[double, ndim=1, mode="c"] f0 not None):

    cdef np.ndarray[double, ndim=1, mode="c"] refined_f0
    refined_f0 = np.copy(f0)
    f0_length = len(f0)
    x_length = len(x)

    StoneMask(&x[0], x_length, fs, &time_axis[0], &f0[0], f0_length, &refined_f0[0])
    return refined_f0

def star(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period,
         np.ndarray[double, ndim=1, mode="c"] time_axis not None,
         np.ndarray[double, ndim=1, mode="c"] f0 not None):

    x_length = len(x)

    cdef int fft_size = GetFFTSizeForCheapTrick(fs)
    cdef int f0_length = len(f0)

    cdef double[:,::1] spectrogram = np.zeros((f0_length,fft_size/2+1))
    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef double **cpp_spectrogram = <double**> (<void*> &tmp[0])
    cdef np.intp_t i
    for i in range(f0_length):
        cpp_spectrogram[i] = &spectrogram[i,0]

    Star(&x[0], x_length, fs, &time_axis[0], &f0[0], f0_length, cpp_spectrogram)

    return np.array(spectrogram, dtype=np.float64)

def cheaptrick(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period,
               np.ndarray[double, ndim=1, mode="c"] time_axis not None,
               np.ndarray[double, ndim=1, mode="c"] f0 not None):

    cdef int x_length = len(x)
    cdef int f0_length = len(f0)
    cdef int fft_size = GetFFTSizeForCheapTrick(fs)

    cdef double[:,::1] spectrogram = np.zeros((f0_length,fft_size/2+1))
    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef double **cpp_spectrogram = <double**> (<void*> &tmp[0])
    cdef np.intp_t i
    for i in range(f0_length):
        cpp_spectrogram[i] = &spectrogram[i,0]
    CheapTrick(&x[0], x_length, fs, &time_axis[0], &f0[0], f0_length, cpp_spectrogram)
    return np.array(spectrogram, dtype=np.float64)

def platinum(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period,
             np.ndarray[double, ndim=1, mode="c"] time_axis not None,
             np.ndarray[double, ndim=1, mode="c"] f0 not None,
             np.ndarray[double, ndim=2, mode="c"] np_spectrogram not None):

    cdef int x_length = len(x)
    cdef int f0_length = len(f0)
    cdef int fft_size = GetFFTSizeForCheapTrick(fs)

    cdef double[:,::1] spectrogram = np_spectrogram
    cdef double[:,::1] residual = np.zeros((f0_length,fft_size+1))

    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef np.intp_t[:] tmp2 = np.zeros(f0_length, dtype=np.intp)

    cdef double **cpp_spectrogram = <double**> (<void*> &tmp[0])
    cdef double **cpp_residual = <double**> (<void*> &tmp2[0])

    cdef np.intp_t i
    for i in range(f0_length):
        cpp_spectrogram[i] = &spectrogram[i,0]
        cpp_residual[i] = &residual[i,0]

    Platinum(&x[0], x_length, fs, &time_axis[0], &f0[0], f0_length,
             cpp_spectrogram, fft_size, cpp_residual)
    return np.array(residual, dtype=np.float64)

def synthesis(int fs, double period,
              np.ndarray[double, ndim=1, mode="c"] f0 not None,
              np.ndarray[double, ndim=2, mode="c"] np_spectrogram not None,
              np.ndarray[double, ndim=2, mode="c"] np_residual not None,
              int y_length):

    cdef int f0_length = len(f0)
    cdef int fft_size = GetFFTSizeForCheapTrick(fs)
    cdef np.ndarray[double, ndim=1, mode="c"] y
    y = np.zeros(y_length, dtype = np.dtype('float64'))

    cdef double[:,::1] spectrogram = np_spectrogram
    cdef double[:,::1] residual = np_residual
    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef np.intp_t[:] tmp2 = np.zeros(f0_length, dtype=np.intp)

    cdef double **cpp_spectrogram = <double**> (<void*> &tmp[0])
    cdef double **cpp_residual = <double**> (<void*> &tmp2[0])
    cdef np.intp_t i
    for i in range(f0_length):
        cpp_spectrogram[i] = &spectrogram[i,0]
        cpp_residual[i] = &residual[i,0]

    Synthesis( &f0[0], f0_length, cpp_spectrogram,cpp_residual, fft_size, period, fs, y_length, &y[0])
    return y

def aperiodicityratio(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period,
                      np.ndarray[double, ndim=1, mode="c"] time_axis not None,
                      np.ndarray[double, ndim=1, mode="c"] f0 not None):

    cdef int x_length = len(x)
    cdef int f0_length = len(f0)
    cdef int fft_size = GetFFTSizeForCheapTrick(fs)

    cdef double[:,::1] aperiodicity = np.zeros((f0_length,fft_size/2+1))
    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef double **cpp_aperiodicity = <double**> (<void*> &tmp[0])
    cdef np.intp_t i
    for i in range(f0_length):
        cpp_aperiodicity[i] = &aperiodicity[i,0]

    AperiodicityRatio(&x[0], x_length, fs, &f0[0], f0_length,  &time_axis[0], fft_size, cpp_aperiodicity)
    return np.array(aperiodicity, dtype=np.float64)

def synthesis_from_aperiodicity(int fs, double period,
              np.ndarray[double, ndim=1, mode="c"] f0 not None,
              np.ndarray[double, ndim=2, mode="c"] np_spectrogram not None,
              np.ndarray[double, ndim=2, mode="c"] np_aperiodicity not None,
              int y_length):

    cdef int f0_length = len(f0)
    cdef int fft_size = GetFFTSizeForCheapTrick(fs)

    cdef np.ndarray[double, ndim=1, mode="c"] y
    y = np.zeros(y_length, dtype = np.dtype('float64'))

    cdef double[:,::1] spectrogram = np_spectrogram
    cdef double[:,::1] aperiodicity = np_aperiodicity
    cdef np.intp_t[:] tmp = np.zeros(f0_length, dtype=np.intp)
    cdef np.intp_t[:] tmp2 = np.zeros(f0_length, dtype=np.intp)
    cdef double **cpp_spectrogram = <double**> (<void*> &tmp[0])
    cdef double **cpp_aperiodicity = <double**> (<void*> &tmp2[0])
    cdef np.intp_t i
    for i in range(f0_length):
        cpp_spectrogram[i] = &spectrogram[i,0]
        cpp_aperiodicity[i] = &aperiodicity[i,0]

    SynthesisFromAperiodicity(&f0[0], f0_length, cpp_spectrogram, cpp_aperiodicity, fft_size, period, fs, y_length, &y[0])
    return y

class pyDioOption:
    def __init__(self, f0_floor, f0_ceil, channels_in_octave, frame_period, speed):
        cdef DioOption option
        InitializeDioOption(&option)
        option.f0_floor = f0_floor
        option.f0_ceil = f0_ceil
        option.channels_in_octave = channels_in_octave
        option.frame_period = frame_period
        option.speed = speed
        self.option = option

