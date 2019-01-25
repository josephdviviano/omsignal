#!/usr/bin/env python

import utils
import visualize
import experiments

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
