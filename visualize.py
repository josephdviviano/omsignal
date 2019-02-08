"""
Generate visualizations from the dataloaders.
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os
import utils

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG = utils.read_config()
PKG_PATH = os.path.dirname(os.path.abspath(__file__))

def get_samples(data, n):
    """
    Gets n X, y samples from input dataloader and returns them as a numpy array.
    """
    X, y = [], []

    for i in range(n):

        # Draw a random sample.
        idx = np.random.randint(0, 160)
        sample = data[idx]
        X.append(sample[0].numpy())
        y.append(sample[1].numpy())

    X = np.vstack(X)
    y = np.vstack(y)

    return(X, y)


def pca(data, n=1000, name='pca'):

    ts_len = CONFIG['models']['tspec']['ts_len']
    X, y = get_samples(data, n)
    X = X[:, :ts_len]

    mdl = PCA(n_components=2)
    mdl.fit(X)
    emb = mdl.transform(X)

    plt.scatter(emb[:, 0], emb[:, 1], c=y[:, -1])
    plt.savefig(os.path.join(PKG_PATH, 'img/{}.jpg'.format(name)))
    plt.close()


def timeseries(data, name='timeseries'):

    ts_len = CONFIG['models']['tspec']['ts_len']
    X, y = get_samples(data, 10)
    X = X[:, :ts_len]

    fig = plt.figure()
    for i in range(10):
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(X[i, :].T)

    plt.savefig(os.path.join(PKG_PATH, 'img/{}.jpg'.format(name)))
    plt.close()


def spectra(data, n=1000, name='spec', log=False):
    """Plot of the mean spectra currently input into model."""
    ts_len = CONFIG['models']['tspec']['ts_len']
    spec_len = CONFIG['models']['tspec']['spec_len']

    X, y = get_samples(data, n)
    X = X[:, ts_len:]

    mean = np.mean(X, axis=0)
    error = np.std(X, axis=0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(spec_len), mean, 'k-')
    ax1.fill_between(np.arange(spec_len), mean-error, mean+error, alpha=0.5)

    if log:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    plt.savefig(os.path.join(PKG_PATH, 'img/{}.jpg'.format(name)))
    ax1.cla()
    plt.close()


def tsne(data, name='tsne'):

    mdl = TSNE(n_components=2, random_state=0,
            perplexity=5, learning_rate=1, n_iter=10000)

    emb = mdl.fit_transform(data.X)

    plt.scatter(emb[:, 0], emb[:, 1], c=data.y[:, -1])
    plt.savefig(os.path.join(PKG_PATH, 'img/{}.jpg'.format(name)))
    plt.close()


def training(results):

    # Plot loss.
    plt.plot(results['train']['losses'])
    plt.plot(results['valid']['losses'])
    plt.axvline(x=results['best_epoch']-1, color='grey', linestyle='dashed')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(PKG_PATH, 'img/training_loss.jpg'))
    plt.close()

    # Plot scores.
    fig, ax = plt.subplots()
    colors =  ['red', 'blue', 'green', 'darkorange', 'black']
    ax.set_prop_cycle('color', colors)
    plt.plot(results['train']['scores'])
    plt.plot(results['valid']['scores'], linestyle='dashed')

    plt.axvline(x=results['best_epoch']-1, color='grey', linestyle='dashed')

    plt.legend(['Training PR Mean',
                'Training RT Mean',
                'Training RR Std',
                'Training ID Recall',
                'Training Total Score',
                'Validation PR Mean',
                'Validation RT Mean',
                'Validation RR Std',
                'Validation ID Recall',
                'Validation Total Score'])
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Scores')
    plt.savefig(os.path.join(PKG_PATH, 'img/training_scores.jpg'))
    plt.close()


