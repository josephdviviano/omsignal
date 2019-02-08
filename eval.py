#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import torch

from utils import read_memfile, read_config, Data
from experiments import evalu_loop


CONFIG = load_config()

def eval_model(dataset_file, model_filename):
    """
    """
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load your best model.
    if model_filename:
        model_filename = Path(model_filename)
        model = torch.load(model_filename, map_location=device)

        print("\nLoading model from", model_filename.absolute())

    if model:

        load_args = {
            'batch_size': CONFIG['dataloader']['batch_size'],
            'num_workers': CONFIG['dataloader']['num_workers']}

        data = read_memfile(dataset_file, shape=(160, 3754), dtype='float32')
        data = {'X': data[:, :3750], 'y': data[:, 3750:]}
        data = Data(precomputed=data, augmentation=False)
        dataloader = torch.utils.data.DataLoader(data, **load_args)

        # Generate predictions with model.
        results = evalu_loop(model, dataloader, return_preds=True)
        y_pred = results['preds']

        # Convert ID column back to original format.
        y_pred[:, -1] = data.ymap.inverse_transform(y_pred[:, -1])


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
    model_filename = "/home/user25/code/omsignal/models/best_tspec_model_080219_11h41.pt"

    # DO NOT MODIFY
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_file, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    write_memfile(results_fname, y_pred)

