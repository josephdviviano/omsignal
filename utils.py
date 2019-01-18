"""
Utilities for handling data.
"""
import numpy as np
import os
import yaml
from sklearn.preprocessing import LabelEncoder


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

    datasets = {'train': train, 'valid': valid}

    # y_map is fit to the final column of the data, which represents ID
    y_map = LabelEncoder()
    y_map.fit(datasets['train'][:, -1])

    return(datasets, y_map)


def convert_y(datasets, y_map):
    """
    y_map is a LabelEncoder instance fit to the training data.
    datasets is a dictionary containing the training and validation data.

    Will automatically flip the ID column of the data  between original labels
    (max=42) and transformed labels (max=31).
    """

    # handle type conversion explicitly for compatibility with y_map
    ids_train = datasets['train'][:, -1].astype(np.int)
    ids_valid = datasets['valid'][:, -1].astype(np.int)

    # Labels are in original format, transform to model-friendly format.

    if np.max(ids_train) == 42:
        datasets['train'][:, -1] = y_map.transform(ids_train)
        datasets['valid'][:, -1] = y_map.transform(ids_valid)

    # Labels are in model-friendly format, transform to original format.
    elif np.max(ids_train) == 31:
        datasets['train'][:, -1] = y_map.inverse_transform(ids_train)
        datasets['valid'][:, -1] = y_map.inverse_transform(ids_valid)

    # Labels have been tampered with, this is bad.
    else:
        raise Exception('Training labels have an illegal max={}'.format(
            np.max(ids_train))
        )


