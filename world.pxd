# file: world.pxd

cimport world

cdef extern from "aperiodicity.h":
    void AperiodicityRatio(double *x, int x_length, int fs, double *f0,
    int f0_length, double *time_axis, int fft_size, double **aperiodicity)

cdef extern from "platinum.h":
    void Platinum(double *x, int x_length, int fs, double *time_axis, double *f0,
    int f0_length, double **spectrogram, int fft_size,
    double **residual_spectrogram)

cdef extern from "synthesis.h":
    void Synthesis(double *f0, int f0_length, double **spectrogram,
    double **residual_spectrogram, int fft_size, double frame_period, int fs,
    int y_length, double *y)

cdef extern from "star.h":
    void Star(double *x, int x_length, int fs, double *time_axis, double *f0,
    int f0_length, double **spectrogram)

cdef extern from "stonemask.h":
    void StoneMask(double *x, int x_length, int fs, double *time_axis, double *f0,
    int f0_length, double *refined_f0)

cdef extern from "synthesisfromaperiodicity.h":
    void SynthesisFromAperiodicity(double *f0, int f0_length, double **spectrogram,
    double **aperiodicity, int fft_size, double frame_period, int fs,
    int y_length, double *y)

cdef extern from "matlabfunctions.h":
    double *wavread(char* filename, int *fs, int *nbit, int *wav_length)
    void wavwrite(double *x, int x_length, int fs, int nbit, char *filename)

cdef extern from "dio.h":
    int GetSamplesForDIO(int fs, int x_length, double frame_period)

    void InitializeDioOption(DioOption *option)

    void Dio(double *x, int x_length, int fs, const DioOption option,
      double *time_axis, double *f0)

    void DioByOptPtr(double *x, int x_length, int fs, const DioOption *option,
      double *time_axis, double *f0)

    ctypedef struct DioOption:
            double f0_floor
            double f0_ceil
            double channels_in_octave
            double frame_period
            int speed

cdef extern from "cheaptrick.h":
    void CheapTrick(double *x, int x_length, int fs, double *time_axis, double *f0,
  int f0_length, double **spectrogram)
    int GetFFTSizeForCheapTrick(int fs)

#cdef class DioOption:
#	cdef wrap1.DioOption* _option
#	self._option = _option
#cdef cqueue.Queue* _c_queue
