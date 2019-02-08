#!/usr/bin/env python
"""
Trains model, saves trained models and visualization.
"""
import datetime
import logging
import pickle
import torch
import os

import experiments
import utils
import visualize

PKG_PATH = os.path.dirname(os.path.abspath(__file__))

# Adds a simple logger.
TSTAMP = datetime.datetime.now().strftime("%d%m%y_%Hh%M")
LOGNAME = os.path.join(PKG_PATH, 'logs/train_{}.log'.format(TSTAMP))
logging.basicConfig(filename=LOGNAME, level=logging.INFO)
LOGGER = logging.getLogger('train')

def main():

    # Run model.
    model, results = experiments.tspec()

    # Plot training curves.
    visualize.training(results)

    # Save model.
    torch.save(model,
        os.path.join(PKG_PATH,
            'models/best_tspec_model_{}.pt'.format(TSTAMP)))

    # Save results.
    utils.write_results(results,
        os.path.join(PKG_PATH,
            'models/best_tspec_results_{}.pkl'.format(TSTAMP)))

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

