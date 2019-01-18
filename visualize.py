
from sklearn.decomposition import PCA
from utils import load_data, convert_y
import os

# Must switch backend to Agg to be compatible with the queue/singularity.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_pca():

    data, ymap = load_data()
    convert_y(data, ymap)

    mdl = PCA(n_components=2)
    mdl.fit(data['train']['X'])



