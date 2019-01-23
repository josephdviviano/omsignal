#!/usr/bin/env python

import utils
import visualize


def main():

    preproc = utils.Preprocessor()
    train_data = utils.Data(augmentation=True)
    valid_data = utils.Data(train=False, augmentation=True)

    print('__getitem__: {}'.format(train_data[0]))

    # convert to numpy for data visualization
    train_data.to_numpy()
    valid_data.to_numpy()
    visualize.plot_spectra(train_data)
    visualize.pca(data)


if __name__ == "__main__":
    main()
