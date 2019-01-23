#!/usr/bin/env python

import utils
import visualize


def main():

    preproc = utils.Preprocessor()
    data = utils.Data()

    # converts y to model-friendly format
    data.convert_y()

    # data to pytorch format (for preprocessing)
    data.to_torch()

    # data is preprocessed (see utils.Preprocessor)
    data.preprocess()

    # data converted to numpy format (for PCA)
    data.to_numpy()

    # generates a PCA embedding of the data
    visualize.pca(data)

    #generate a TSNE embedding of the data
    visualize.tsne(data)


if __name__ == "__main__":
    main()
