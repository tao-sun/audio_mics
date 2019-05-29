import sys
import math

import numpy as np
import numpy.matlib as matlab
from scipy.io import wavfile
import scipy.signal as sps


def resample(x, rate, new_rate):
    number_of_samples = int(round(len(x) * float(new_rate) / rate))
    print('num of samples: %d' % number_of_samples)
    x = sps.resample(x, number_of_samples)
    return x


def thirdoct(fs, N_fft, num_bands, mn):
    """
    Calculate 1/3 octave band matrix.
    :param fs: samplerate
    :param N_fft: FFT size
    :param num_bands: number of bands
    :param mn: center frequency of first 1/3 octave band
    :return:
        A: octave band matrix
        CF: center frequencies
    """

    f = np.linspace(0, fs, N_fft+1)
    f = f[0:(N_fft / 2 + 1)]
    k = np.array(range(0, num_bands))
    cf = np.power(2, (k/3.0)) * mn
    fl = np.sqrt(np.multiply(np.power(2, (k / 3.0)) * mn, np.float_power(2, ((k - 1) / 3.0)) * mn))
    fr = np.sqrt(np.multiply(np.power(2, (k / 3.0)) * mn, np.power(2, ((k + 1) / 3.0)) * mn))
    A = np.zeros((num_bands, len(f)))

    for i in range(0, len(cf)):
        c = np.power((f - fl[i]), 2)
        a, b = np.amin(c), np.argmin(c)
        fl[i] = f[b]
        fl_ii = b

        c = np.power((f - fr[i]), 2)
        a, b = np.amin(c), np.argmin(c)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1

    rnk = np.sum(A, 1)
    rnk_and = np.logical_and(rnk[1:] >= rnk[0:-1], rnk[1:] != 0)
    num_bands = np.argwhere(rnk_and != 0)[-1][0] + 2
    A = A[0:num_bands, :]
    cf = cf[0:num_bands]

    return A


def remove_silent_frames(x, y, dyn_range, N, K):
    """
        x and y are segmented with frame-length N and overlap K, where the maximum energy
        of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
        reconstructed signals, excluding the frames, where the energy of a frame
        of X is smaller than X_MAX-RANGE.
    """

    x = np.transpose(x)
    y = np.transpose(y)

    frames = range(0, len(x)-N, K)
    w = np.hanning(N+1)[1:]
    msk = np.zeros(len(frames))

    for j in range(0, len(frames)):
        jj = range(frames[j], frames[j] + N)
        msk[j] = 20 * np.log10(np.linalg.norm(x[jj] * w) / math.sqrt(N))

    msk = (msk - np.amax(msk) + dyn_range) > 0
    count = 0

    x_sil = np.zeros(len(x))
    y_sil = np.zeros(len(y))

    # for i in range(len(frames)):
    #     if msk[i]:
    #         for jj_i in range(frames[i], frames[i] + N):
    #             if jj_i % 1000 == 0:
    #                 print(x[jj_i], y[jj_i])

    for j in range(0, len(frames)):
        if msk[j]:
            jj_i = range(frames[j], frames[j] + N)
            jj_o = range(frames[count], frames[count] + N)
            x_sil[jj_o] = x_sil[jj_o] + x[jj_i] * w
            y_sil[jj_o] = y_sil[jj_o] + y[jj_i] * w
            count = count + 1

    x_sil = x_sil[:jj_o[-1]]
    y_sil = y_sil[:jj_o[-1]]

    # for i in range(0, len(x_sil), 1000):
    #     print(x_sil[i], y_sil[i])
    return x_sil, y_sil


def stdft(x, N, K, N_fft):
    """
    Returns the short-time hanning-windowed dft of X with frame-size N,
    overlap K and DFT size N_FFT. The columns and rows of X_STDFT denote
    the frame-index and dft-bin index, respectively.
    """
    frames = range(0, len(x) - N, K)
    x_stdft = np.zeros((len(frames), N_fft))

    w = np.hanning(N+1)[1:]
    x = np.transpose(x)

    for i in range(len(frames)):
        x_stdft[i] = np.fft.fft(x[frames[i]:(frames[i]+N)] * w, N_fft)

    return x_stdft


def taa_corr(x, y):
    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0, 1]


def stoi(clean_file, enhd_file):
    s_rate1, x = wavfile.read(clean_file)
    s_rate2, y = wavfile.read(enhd_file)

    assert (s_rate1 == s_rate2 and len(x) == len(y)), "The two file must have the same frame rate and length."

    fs_signal = s_rate1

    # initialization
    x = np.transpose(x)  # clean speech column vector
    y = np.transpose(y)  # processed speech column vector

    fs = 10000  # sample rate of proposed intelligibility measure
    N_frame = 256  # window support
    K = 512  # FFT_size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz.
    H = thirdoct(fs, K, J, mn)  # Get 1/3 octave band matrix
    print('H shape: {0}'.format(H.shape))
    N = 30  # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15  # lower SDR-bound
    dyn_range = 40  # speech dynamic range

    # resample signals if other samplerate is used than fs
    if fs_signal != fs:
        x = resample(x, fs_signal, fs)
        y = resample(y, fs_signal, fs)

    # remove silent frames
    x, y = remove_silent_frames(x, y, dyn_range, N_frame, N_frame / 2)

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x, N_frame, N_frame/2, K)  # apply short-time DFT to clean speech
    print('original x_hat shape: {0}'.format(x_hat.shape))
    y_hat = stdft(y, N_frame, N_frame/2, K)  # apply short-time DFT to processed speech

    x_hat = np.transpose(x_hat[:, 0:(K/2+1)])  # take clean single-sided spectrum
    print('x_hat shape: {0}'.format(x_hat.shape))
    y_hat = np.transpose(y_hat[:, 0:(K/2+1)])  # take processed single-sided spectrum

    X = np.sqrt(np.dot(H, np.power(np.absolute(x_hat), 2)))  # apply 1/3 octave bands as described in Eq.(1) [1]
    print('X shape: {0}'.format(X.shape))
    Y = np.sqrt(np.dot(H, np.power(np.absolute(y_hat), 2)))

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm = np.zeros((J, len(range(N, X.shape[1]+1))))  # init memory for intermediate intelligibility measure
    c = math.pow(10, -Beta/20.0)  # constant for clipping procedure
    print('c is %f' % c)

    for m in range(N, X.shape[1]+1):
        X_seg = X[:, (m-N):m]  # region with length N of clean TF-units for all j
        Y_seg = Y[:, (m-N):m]  # region with length N of processed TF-units for all j

        # obtain scale facto for normalizing processed TF-region for all j
        alpha = np.sqrt(np.sum(np.power(X_seg, 2), 1) / np.sum(np.power(Y_seg, 2), 1))
        # obtain \alpha * Y_j(n) from Eq.(2) [1]
        aY_seg = Y_seg * matlab.repmat(alpha.reshape(alpha.shape[0], 1), 1, N)

        for j in range(J):
            # apply clipping from Eq.(3)
            Y_prime = np.minimum(aY_seg[j, :], X_seg[j, :] + X_seg[j, :] * c)
            # obtain correlation coeffecient from Eq.(4) [1]
            d_interm[j, m-N] = taa_corr(np.transpose(X_seg[j, :]), Y_prime)

    d = np.mean(d_interm)  # combine all intermediate intelligibility measures as in Eq.(4)[1]
    return d


if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]

    x = stoi(f1, f2)
    print(x)
    # s_rate1, x1 = wavfile.read(f1)
    # s_rate2, y1 = wavfile.read(f2)
    #
    # print(s_rate1, s_rate2)
    # print(len(x1)/float(len(x)), len(y1)/float(len(y)))
    #
    # fs = 10000  # sample rate of proposed intelligibility measure
    # N_frame = 256  # window support
    # K = 512  # FFT_size
    # J = 15  # Number of 1/3 octave bands
    # mn = 150  # Center frequency of first 1/3 octave band in Hz.
    # H, a = thirdoct(fs, K, J, mn)  # Get 1/3 octave band matrix
    # # print(H[0, 0:30])
