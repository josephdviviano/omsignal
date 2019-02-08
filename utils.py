"""
Utilities for handling data.
"""

from colorednoise import powerlaw_psd_gaussian
from copy import copy
from scipy import signal
from scipy.stats import kendalltau
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import models

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# seed for testing purposes
np.random.seed(seed=12345678)


def read_config():
    """Returns the config.yml file as a dictionary."""
    with open('config.yml', 'r') as fname:
        return(yaml.load(fname))


def write_results(filename):
    """Serialize a dictionary containing best model, performance, and ymap."""
    with open(filename, 'wb') as hdl:
        pickle.dump(results, hdl, protocol=pickle.HIGHEST_PROTOCOL)


def read_results(filename):
    """Load a dictionary containing best model, performance, and ymap."""
    with open(filename, 'rb') as hdl:
        results = pickle.load(filename)

    return(results)


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
    Write a numpy array 'data' into a binary data file 'filename'.
    """
    shape = data.shape
    dtype = data.dtype
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    fp[:] = data[:]
    del fp


def get_shuffled_data(test_p=0.5):
    """
    Mixes samples from day 1 (train) and 2 (valid), since there might be
    important distributional differences between them.

    By default, the resulting splits are 50/50, like in the original data.
    If test_p < 0.5, more data is allocated to the training set.
    """
    CONFIG = read_config()

    # Create merged dataset.
    train_data = read_memfile(
            os.path.join(CONFIG['data'], 'MILA_TrainLabeledData.dat'),
            shape=(160, 3754), dtype='float32')

    valid_data = read_memfile(
            os.path.join(CONFIG['data'], 'MILA_ValidationLabeledData.dat'),
            shape=(160, 3754), dtype='float32')

    data = np.vstack([train_data, valid_data])

    # Split data shuffled across both datasets (day1=train, day2=valid).
    ids = data[:, -1]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
    for train_idx, valid_idx in sss.split(data, ids):
        train_data, valid_data = data[train_idx], data[valid_idx]

    # Return data as a giant dictionary.
    data = {'train': {'X': train_data[:, :3750], 'y': train_data[:, 3750:]},
            'valid': {'X': valid_data[:, :3750], 'y': valid_data[:, 3750:]}}

    return(data)


class Data(Dataset):
    """
    Object for handling the data, as well as augmentations.
    """

    def __init__(self, precomputed={}, train=True, augmentation=False):
        """
        Loads the data into memory, as well as a method for switching
        user IDs from the original format into a model-friendly format.
        """

        CONFIG = read_config()

        # precomputed can contain shuffled data from get_shuffled_data().
        # Otherwise we just load directly from disk.
        if not precomputed:
            if train:
                filename = 'MILA_TrainLabeledData.dat'
            else:
                filename = 'MILA_ValidationLabeledData.dat'

            data = read_memfile(
                os.path.join(CONFIG['data'], filename),
                shape=(160, 3754), dtype='float32')

            self.X = data[:, :3750]
            self.y = data[:, 3750:]

        else:
            try:
                self.X = precomputed['X']
                self.y = precomputed['y']
            except:
                raise Exception('precomputed is not a valid input dict!')

        # Keep track of the current data format.
        self.noise_gain = CONFIG['preprocessing']['noise_gain']
        self.format = 'numpy'
        self.augmentation = augmentation

        # used for preprocessing
        self.preproc = Preprocessor()

        # ymap is fit to the final column of y, which represents ID.
        self.ymap = LabelEncoder()
        self.ymap.fit(self.y[:, -1])

        # y is automatically converted to model-friendly format
        self.convert_y()

    def __len__(self):
        """Returns the number of samples."""
        return(len(self.X))

    def __getitem__(self, idx):
        """
        Returns a single preprocessed (and optionally augmented) ECG time
        series and associated labels from the dataset.
        1. Optionally augments the data (rotations, added noise).
        2. Calculates the PSD (power spectral density) of the signal and
           concatenates these features to the ECG timeseries.
        3. TODO: add spectrogram?
        """

        X = self.X[idx, :]
        y = self.y[idx, :]

        if self.format == 'numpy':
            X = torch.Tensor(X)
            y = torch.Tensor(y)

        X = self.preproc.forward(X)

        if self.augmentation:
            X = self.augment(X)

        # All samples have (normalized) spectra computed regardless of
        # augmentation.
        _, pxx = signal.periodogram(X)
        spectra = torch.Tensor(pxx)
        #spectra /= torch.std(spectra)

        # See config.yml for ts_len=X.size(), spec_len=spectra.size()
        # which is used by the model to split X appropriately.
        X = torch.cat([X, spectra])

        return(X, y)

    def to_torch(self):
        """Converts all data to pytorch format."""
        if self.format == 'torch':
            return

        self.X = torch.Tensor(self.X)
        self.y = torch.Tensor(self.y)
        self.format = 'torch'

    def to_numpy(self):
        """Converts data to numpy format."""
        if self.format == 'numpy':
            return

        self.X = self.X.numpy()
        self.y = self.y.numpy()
        self.format = 'numpy'

    def convert_y(self):
        """
        y_map is a LabelEncoder instance fit to the training data.
        datasets is a dictionary containing the training and validation data.

        Will automatically flip the ID column of the data  between original
        labels (max=42) and transformed labels (max=31).
        """
        if self.format == 'torch':
            raise Exception('convert_y must be run with data in numpy format')

        # Handle type conversion explicitly for compatibility with ymap
        ids = self.y[:, -1].astype(np.int)

        # Labels are in original format, transform to model-friendly format.
        if np.max(ids) == 42:
            self.y[:, -1] = self.ymap.transform(ids)

        # Labels are in model-friendly format, transform to original format.
        elif np.max(ids) == 31:
            self.y[:, -1] = self.ymap.inverse_transform(ids)

        # Labels have been tampered with, this is bad.
        else:
            raise Exception('Training labels have an illegal max={}'.format(
                np.max(ids)))

    def augment(self, sample):
        """
        Returns an augmented sample from X.
        1. Adds pink noise (scaled by noise_gain in config) to all timepoints.
        2. Rotates the signal by wrapping values from the start to the end.
        """
        n = len(sample)

        # Add coloured noise into the data.
        noise = powerlaw_psd_gaussian(1, len(sample)) # 1 = pink noise
        sample += torch.Tensor(noise)*self.noise_gain

        # Rotate samples.
        rotated_sample = torch.zeros(n)
        start_idx = np.random.randint(0, n, 1)[0]
        start_len = n - start_idx
        rotated_sample[:start_len] = sample[start_idx:]
        rotated_sample[start_len:] = sample[:start_idx]

        return(rotated_sample)


class Preprocessor(nn.Module):

    def __init__(
            self, ma_win=2, mv_win=4, num_samples_per_second=125):
        """
        ma_win: window size (secs) for moving average baseline wander removal
        mv_win: window size (secs) for moving average RMS normalization
        """
        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd
        self.ma_kernel = (ma_win * num_samples_per_second)+1

        # Kernel size to use for moving average normalization: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd
        self.mv_kernel = (mv_win * num_samples_per_second)+2

    def forward(self, x):

        eps = 1e-5
        with torch.no_grad():

            # Add two dummy dimensions for (batch_size, sample) respectively
            # i.e., x is (batch_size=1, sample=1, timeseries=n_elements).
            x = x.unsqueeze(0).unsqueeze(0)

            # Remove total mean and standard deviation.
            x = (x - torch.mean(x, dim=2, keepdim=True)) / (torch.std(x,
                dim=2, keepdim=True) + eps)

            # Moving average baseline wander removal.
            x = x - F.avg_pool1d(
                x, kernel_size=self.ma_kernel,
                stride=1, padding=(self.ma_kernel - 1) // 2
            )

            # Moving RMS normalization.
            x = x / (torch.sqrt(F.avg_pool1d(torch.pow(x, 2),
                kernel_size=self.mv_kernel-1, stride=1,
                padding=(self.mv_kernel-1) // 2)) + eps
            )

        # Get rid of singleton dimension.
        x = x.squeeze()

        # Don't backpropagate further.
        x = x.detach().contiguous()

        return(x)


def assert_score(x):
    """
    Score 'x' should be a 1-D numpy array containing np.int32s or float32s.
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    assert x.dtype in [np.int32, np.float32]


def score_performance(prMean_pred, prMean_true, rtMean_pred, rtMean_true,
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
    id_recall = recall_score(ecgId_true, ecgId_pred, average='macro')
    adjustement = 1.0 / len(np.unique(ecgId_true))
    id_recall = (id_recall - adjustement) / (1 - adjustement)
    if id_recall < 0:
        id_recall = 0.0

    # Compute Kendall correlation coefficients for regression tasks.
    # Any coefficients worse than chance will be clipped to zero.
    rr_std_score, _ = kendalltau(rrStd_pred, rrStd_true)
    if rr_std_score < 0:
        rr_std_score = 0.0

    pr_mean_score, _ = kendalltau(prMean_pred, prMean_true)
    if pr_mean_score < 0:
        pr_mean_score = 0.0

    rt_mean_score, _ = kendalltau(rtMean_pred, rtMean_true)
    if rt_mean_score < 0:
        rt_mean_score = 0.0

    # Compute the final performance score as the geometric mean of the
    # individual task performances. A high geometric mean ensures that
    # there are no tasks with very poor performance that are masked by good
    # performance on the other tasks. If any task has chance performance
    # or worse, the overall performance will be zero. If all tasks are
    # perfectly solved, the overall performance will be 1.
    total_score = np.power(
        rr_std_score * pr_mean_score * rt_mean_score * id_recall, 0.25)

    return(total_score, pr_mean_score, rt_mean_score, rr_std_score, id_recall)


