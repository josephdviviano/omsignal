#!/usr/bin/env python
"""
Trains model, saves trained models and visualization.
"""
import datetime
import logging

import experiments
import utils
import visualize

# Adds a simple logger.
TSTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(filename='logs/train_{}.log'.format(TSTAMP), level=logging.INFO)
LOGGER = logging.getLogger('train')

def main():

    # Run model.
    results = experiments.tspec()

    # Visualizations using non-shuffled data.
    train_data = utils.Data(train=True, augmentation=True)
    valid_data = utils.Data(train=False, augmentation=False)

    visualize.spectra(train_data, log=False, name='spectra_train')
    visualize.spectra(valid_data, log=False, name='spectra_valid')
    visualize.timeseries(train_data, name='timeseries_train')
    visualize.timeseries(valid_data, name='timeseries_valid')
    visualize.pca(train_data)
    visualize.tsne(train_data)

if __name__ == "__main__":
    main()


