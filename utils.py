"""
Utilities for handling data.
"""

from scipy import signal
from scipy.stats import kendalltau
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loads the configuration file as a global variable.
with open('config.yml', 'r') as fname:
    CONFIG = yaml.load(fname)


def read_memfile(filename, shape, dtype='float32'):
    """
    Read binary data and return as a numpy array.
    """

    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    data = np.zeros(shape=shape, dtype=dtype)
    data[:] = fp[:]
    del fp

    return(data)


def write_memfile(data, filename):
    """
    Write a numpy array 'data' into a binary  data file specified by
    'filename'.
    """

    shape = data.shape
    dtype = data.dtype
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    fp[:] = data[:]
    del fp


def load_data():
    """Returns a dictionary containing the train and valid datasets."""

    train = read_memfile(
        os.path.join(CONFIG['data'], 'MILA_TrainLabeledData.dat'),
        shape=(160, 3754), dtype='float32'
    )

    valid = read_memfile(
        os.path.join(CONFIG['data'], 'MILA_ValidationLabeledData.dat'),
        shape=(160, 3754), dtype='float32'
    )

    train_X = train[:, :3750]
    valid_X = valid[:, :3750]
    train_y = train[:, 3750:]
    valid_y = valid[:, 3750:]

    datasets = {'train': {'X': train_X, 'y': train_y},
                'valid': {'X': valid_X, 'y': valid_y}
    }

    # y_map is fit to the final column of y, which represents ID
    y_map = LabelEncoder()
    y_map.fit(datasets['train']['y'][:, -1])

    return(datasets, y_map)


def convert_y(datasets, y_map):
    """
    y_map is a LabelEncoder instance fit to the training data.
    datasets is a dictionary containing the training and validation data.

    Will automatically flip the ID column of the data  between original labels
    (max=42) and transformed labels (max=31).
    """

    # handle type conversion explicitly for compatibility with y_map
    ids_train = datasets['train']['y'][:, -1].astype(np.int)
    ids_valid = datasets['valid']['y'][:, -1].astype(np.int)

    # Labels are in original format, transform to model-friendly format.
    if np.max(ids_train) == 42:
        datasets['train']['y'][:, -1] = y_map.transform(ids_train)
        datasets['valid']['y'][:, -1] = y_map.transform(ids_valid)

    # Labels are in model-friendly format, transform to original format.
    elif np.max(ids_train) == 31:
        datasets['train']['y'][:, -1] = y_map.inverse_transform(ids_train)
        datasets['valid']['y'][:, -1] = y_map.inverse_transform(ids_valid)

    # Labels have been tampered with, this is bad.
    else:
        raise Exception('Training labels have an illegal max={}'.format(
            np.max(ids_train))
        )


class Preprocessor(nn.Module):

    def __init__(
            self, ma_win=2, mv_win=4, num_samples_per_second=125):
        """
        ma_win: window size (secs) to use for moving average baseline wander removal
        mv_win: window size (secs) to use for moving average RMS normalization
        """

        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd
        self.maKernelSize = (ma_win * num_samples_per_second)+1

        # Kernel size to use for moving average normalization: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd
        self.mvKernelSize = (mv_win * num_samples_per_second)+1

    def forward(self, x):

        with torch.no_grad():

            # Remove window mean and standard deviation
            x = (x - torch.mean(x, dim=2, keepdim=True)) / \
                (torch.std(x, dim=2, keepdim=True) + 0.00001)

            # Moving average baseline wander removal
            x = x - F.avg_pool1d(
                x, kernel_size=self.maKernelSize,
                stride=1, padding=(self.maKernelSize - 1) // 2
            )

            # Moving RMS normalization
            x = x / (
                torch.sqrt(
                    F.avg_pool1d(
                        torch.pow(x, 2),
                        kernel_size=self.mvKernelSize,
                        stride=1, padding=(self.mvKernelSize - 1) // 2
                    )
                ) + 0.00001
            )

        # Don't backpropagate further.
        x = x.detach().contiguous()

        return(x)


def assert_score(x):
    """Score 'x' should be a 1-D numpy array containing np.int32s."""
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    assert x.dtype == np.int32


def scorePerformance(prMean_pred, prMean_true, rtMean_pred, rtMean_true,
    rrStd_pred, rrStd_true, ecgId_pred, ecgId_true):
    """
    Computes the combined multitask performance score.
    The 3 regression tasks are individually scored using Kendalls
    correlation coeffficient.

    The user classification task is scored according to macro averaged
    recall, with an adjustment for chance level. All performances are
    clipped at 0.0, so that zero indicates chance or worse performance,
    and 1.0 indicates perfect performance.

    The individual performances are then combined by taking
    the geometric mean.

    :param prMean_pred: 1D float32 numpy array.
        The predicted average P-R interval duration over the window.
    :param prMean_true: 1D float32 numpy array.
        The true average P-R interval duration over the window.
    :param rtMean_pred: 1D float32 numpy array.
        The predicted average R-T interval duration over the window.
    :param rtMean_true: 1D float32 numpy array.
        The true average R-T interval duration over the window.
    :param rrStd_pred: 1D float32 numpy array.
        The predicted R-R interval duration standard deviation over the window.
    :param rrStd_true: 1D float32 numpy array.
        The true R-R interval duration standard deviation over the window.
    :param ecgId_pred: 1D int32 numpy array.
        The predicted user ID label for each window.
    :param ecgId_true: 1D int32 numpy array.
        The true user ID label for each window.

    :return:
        - The combined performance score on all tasks;
                0.0 means at least one task has chance level performance
                or worse while 1.0 means all tasks are solved perfectly.
        - The individual task performance scores are also returned
    """

    # Input checking.
    assert_score(ecgId_pred)
    assert_score(ecgId_true)
    assert_score(rrStd_pred)
    assert_score(rrStd_true)
    assert_score(prMean_pred)
    assert_score(prMean_true)
    assert_score(rtMean_pred)
    assert_score(rtMean_true)

    assert (len(ecgId_pred) == len(ecgId_true)) \
        and (len(ecgId_pred) == len(prMean_pred)) \
        and (len(ecgId_pred) == len(prMean_true)) \
        and (len(ecgId_pred) == len(rtMean_pred)) \
        and (len(ecgId_pred) == len(rtMean_true)) \
        and (len(ecgId_pred) == len(rrStd_pred)) \
        and (len(ecgId_pred) == len(rrStd_true))

    # Accuracy is computed with macro averaged recall so that accuracy
    # is computed as though the classes were balanced, even if they are not.
    # Note that provided training, validation and testing sets are balanced.
    # Unbalanced classes would only be and issue if a new train/validation
    # split is created.
    # Any accuracy value worse than random chance will be clipped at zero.
    ecgIdAccuracy = recall_score(ecgId_true, ecgId_pred, average='macro')
    adjustementTerm = 1.0 / len(np.unique(ecgId_true))
    ecgIdAccuracy = (ecgIdAccuracy - adjustementTerm) / (1 - adjustementTerm)
    if ecgIdAccuracy < 0:
        ecgIdAccuracy = 0.0

    # Compute Kendall correlation coefficients for regression tasks.
    # Any coefficients worse than chance will be clipped to zero.
    rrStdTau, _ = kendalltau(rrStd_pred, rrStd_true)
    if rrStdTau < 0:
        rrStdTau = 0.0

    prMeanTau, _ = kendalltau(prMean_pred, prMean_true)
    if prMeanTau < 0:
        prMeanTau = 0.0

    rtMeanTau, _ = kendalltau(rtMean_pred, rtMean_true)
    if rtMeanTau < 0:
        rtMeanTau = 0.0

    # Compute the final performance score as the geometric mean of the
    # individual task performances. A high geometric mean ensures that
    # there are no tasks with very poor performance that are masked by good
    # performance on the other tasks. If any task has chance performance
    # or worse, the overall performance will be zero. If all tasks are
    # perfectly solved, the overall performance will be 1.
    combinedPerformanceScore = np.power(
        rrStdTau * prMeanTau * rtMeanTau * ecgIdAccuracy, 0.25)

    return(
        combinedPerformanceScore,
        prMeanTau, rtMeanTau, rrStdTau, ecgIdAccuracy
    )


def make_spectogram(x, lognorm=False, fs=16, nperseg=256, noverlap=None):
    """
    takes a time-series x and returns the spectogram
    input:
           x: time series
           lognorm: bool - log spectrogram or not (default: False)
           fs: float - Sampling frequency of the x time series.
                Defaults to 16.
           nperseg: int - Length of each segment (default: 256).
           noverlap: int, Number of points to overlap between segments.
                  If None, noverlap = nperseg // 8. Defaults to None.
    output :
          f : ndarray - Array of sample frequencies.
          t : ndarray  - Array of segment times.
          Zxx : ndarray - Spectrogram of x.
              By default, the last axis of Zxx corresponds
              to the segment times.
    """

    f, t, Zxx = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap
    )

    if lognorm:
        Zxx = np.abs(Zxx)
        mask = Zxx > 0
        Zxx[mask] = np.log(Zxx[mask])
        Zxx = (Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx))

    return(f, t, Zxx)


def make_fft(x):
    """
    Takes a time-series x and returns the FFT
    input: x
    output : R and I; real and imaginary componenet of the real FFT
    """
    y = np.fft.rfft(x)

    return(np.real(y), np.imag(y))


def score_example():
    prMean_pred = np.random.randn(480).astype(np.float32)
    prMean_true = (np.random.randn(480).astype(
        np.float32) / 10.0) + prMean_pred

    rtMean_pred = np.random.randn(480).astype(np.float32)
    rtMean_true = (np.random.randn(480).astype(
        np.float32) / 10.0) + rtMean_pred

    rrStd_pred = np.random.randn(480).astype(np.float32)
    rrStd_true = (np.random.randn(480).astype(np.float32) / 10.0) + rrStd_pred

    ecgId_pred = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)
    ecgId_true = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)

    print(
        scorePerformance(
            prMean_true, prMean_pred,
            rtMean_true, rtMean_pred,
            rrStd_true, rrStd_pred,
            ecgId_true, ecgId_pred
        )
    )


def plot_ecgfft(x, y):
    """Plots the real and imaginary part of the FFT of an ECG signal."""

    plt.title('ECG FFT')
    plt.plot(x[0, 0, :])
    plt.plot(y[0, 0, :])
    plt.xlabel('Frequency')
    plt.ylabel('FFT')
    plt.legend(['Real', 'Imag'])
    plt.savefig('fft_visual.png')
    plt.close()


def plot_spectrogram(f, t, Zxx):

    plt.title('Spectrogram')
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=1)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig('stft.png')
    plt.close()


if __name__ == '__main__':

    score_example()

    # reads fake ecg data and plots a FFT
    fake_ecg = np.random.randn(3750).astype(np.float32)
    fftr, ffti = make_fft(fake_ecg)
    plot_ecgfft(fftr, ffti)

    # reads fake ecg data and plots a Spectogram
    f, t, Zxx = make_spectogram(fake_ecg, True)
    plot_spectrogram(f, t, Zxx)

