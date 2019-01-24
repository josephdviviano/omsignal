from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import make_fft
import numpy as np
import os

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pca(data, name='pca'):

    mdl = PCA(n_components=2)
    mdl.fit(data.X)
    emb = mdl.transform(data.X)

    plt.scatter(emb[:, 0], emb[:, 1], c=data.y[:, -1])
    plt.savefig('{}.jpg'.format(name))
    plt.close()


def plot_spectra(data, name='spec'):
    mean_spec = np.zeros(1876)
    n = data.X.shape[0]

    for i in range(n):
        spec, _ = make_fft(data.X[0, :])
        mean_spec += np.abs(spec)**2
    mean_spec /= n
    plt.loglog(mean_spec)
    plt.savefig('{}.jpg'.format(name))
    plt.close()



def tsne(data,name='tsne'):
    
    mdl = TSNE(n_components=2, random_state=0,
            perplexity=5, learning_rate=1, n_iter=10000)
    mdl.fit(data.X)
    emb = mdl.transform(data.X)
    
    
    plt.scatter(emb[:, 0], emb[:, 1], c=data.y[:, -1])
    plt.savefig('{}.jpg'.format(name))
    plt.close()
    
     
    import IPython; IPython.embed()


