#!/usr/bin/env python

import utils
import visualize
import experiments
import logging
import datetime


# adds a simple logger
TSTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(filename='logs/train_{}.log'.format(TSTAMP), level=logging.INFO)
LOGGER = logging.getLogger('train')
#LOGHDL = logging.FileHandler('logs/train_{}.log'.format(TSTAMP))
#LOGHDL.setFormatter(
#    logging.Formatter("[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s"))
#LOGGER.addHandler(LOGHDL)
#LOGGER.setLevel(logging.INFO)

def main():

    # run experiment
    #results = experiments.lstm()

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


