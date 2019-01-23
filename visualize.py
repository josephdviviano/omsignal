
from sklearn.decomposition import PCA
import os

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pca(data):

    mdl = PCA(n_components=2)
    mdl.fit(data.train_X)
    train_X_emb = mdl.transform(data.train_X)
    valid_X_emb = mdl.transform(data.valid_X)

    plt.scatter(train_X_emb[:, 0], train_X_emb[:, 1], c=data.train_y[:, -1])
    plt.savefig('pca_train.jpg')
    plt.close()

    plt.scatter(valid_X_emb[:, 0], valid_X_emb[:, 1], c=data.valid_y[:, -1])
    plt.savefig('pca_valid.jpg')
    plt.close()

    import IPython; IPython.embed()



