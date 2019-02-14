#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import torch
import sys

sys.path.append('../')

import utils
import experiments
import models

CUDA = torch.cuda.is_available()

def eval_model(dataset_file, model_filename, results_filename):
    """
    Docstring for eval_model.
    """
    model = None

    # Load your best model.
    if model_filename:

        model_filename = Path(model_filename)
        model = torch.load(model_filename)

        if CUDA:
            model = model.cuda()

        print("\nLoading model from", model_filename.absolute())

    if model:

        N_SUBJ = 160
        data = utils.read_memfile(dataset_file, shape=(N_SUBJ, 3750), dtype='float32')
        results = utils.read_results(results_filename)

        # y is generated as we do not have predictions here.
        fake_y = np.random.randint(1, 32, size=((N_SUBJ, 4)))
        fake_y[0, -1] = 32

        data = {'X': data, 'y': fake_y}
        data = utils.Data(precomputed=data, augmentation=False)

        # Overwrite ymap in Data with one computed using real data.
        data.ymap = results['ymap']

        dataloader = torch.utils.data.DataLoader(data, batch_size=64, num_workers=2)

        # Generate predictions with model.
        results = experiments.evalu_loop(model, dataloader, return_preds=True)
        y_pred = results['preds']

        # Convert ID column back to original format.
        ids = y_pred[:, -1]
        ids = data.ymap.inverse_transform(ids.astype(np.int))
        y_pred[:, -1] = ids

    else:
        print("\nYou did not specify a model, generating dummy data instead!")
        c = 32
        n = 10

        y_pred = np.concatenate(
            [np.random.rand(n, 3), np.random.randint(0, c, (n, 1))],
                axis=1).astype(np.float32)

    return(y_pred)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='',
        help='Absolute path to the evaluation dataset.')
    parser.add_argument("--results_dir", type=str, default='',
        help='Absolute path the results directory.')

    args = parser.parse_args()
    dataset_file = args.dataset
    results_dir = args.results_dir

    # TODO: Ensure permissions.
    group_name = "b1pomt3"
    model_filename = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt3/model/best_model.pth"
    results_filename = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt3/model/best_model_results.pkl"

    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_file, model_filename, results_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 2, "Make sure ndim=2 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    utils.write_memfile(y_pred, results_fname)

