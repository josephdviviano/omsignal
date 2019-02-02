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
    results = experiments.lstm()

    # visualizations
    train_data = utils.Data(augmentation=True)
    valid_data = utils.Data(train=False, augmentation=True)

    visualize.plot_spectra(train_data)
    visualize.pca(train_data)
    visualize.tsne(train_data)


if __name__ == "__main__":
    main()
