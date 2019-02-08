"""
Code for training and evaluating Pytorch models.
"""
from torch.nn.modules.loss import MarginRankingLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import numpy as np
import os
import pprint
import time
import torch
import torch.optim as optim

import models
import utils

CUDA = torch.cuda.is_available()
CONFIG = utils.read_config()
LOGGER = logging.getLogger(os.path.basename(__file__))

def calc_losses(y_hats, y, out_dims):
    """
    Calculate all losses across all prediction tasks.
    Also reformats 'predictions' to be a friendly pytorch tensor for later use.

    TODO: this should be a class?
    """
    reg_loss = MarginRankingLoss()
    clf_loss = CrossEntropyLoss()

    if CUDA:
        reg_loss = reg_loss.cuda()
        clf_loss = clf_loss.cuda()

    losses, predictions = [], []

    for i, out_dim in enumerate(out_dims):
        y_hat = y_hats[i]
        y_tru = y[:, i]

        # Regression case.
        if out_dim == 1:

            # Required for margin ranking loss.
            y_rank = get_paired_ranks(y_tru)
            y1_hat, y2_hat = get_pairs(y_hat)

            losses.append(reg_loss(y1_hat, y2_hat, y_rank))
            predictions.append(y_hat)

        # Classification case.
        elif out_dim > 1:

            # Cross entropy loss.
            losses.append(clf_loss(y_hat, y_tru.long()))
            _, preds = torch.max(y_hat.data, 1)
            predictions.append(preds.float().unsqueeze(1))

    predictions = torch.cat(predictions, dim=1)

    return(losses, predictions)


def get_pairs(y):
    """
    For an input vector y, returns vectors y1 and y2 such that y1-y2 gives all
    unique pairwise subtractions possible in y.
    """
    y = y.cpu()

    n = len(y)
    idx_y2, idx_y1 = np.where(np.tril(np.ones((n, n)), k=-1))
    y1 = y[torch.LongTensor(idx_y1)]
    y2 = y[torch.LongTensor(idx_y2)]

    if CUDA:
        y1 = y1.cuda()
        y2 = y2.cuda()

    return(y1, y2)


def get_paired_ranks(y):
    """
    Generate y_rank (for margin ranking loss). If `y == 1` then it assumed the
    first input should be ranked higher (have a larger value) than the second
    input, and vice-versa for `y == -1`.
    """
    y = y.cpu()

    # Calculates all pairwise subtractions.
    y_rank = y[np.newaxis, :] - y[:, np.newaxis]

    # Edge case where the difference between 2 points is 0.
    y_rank[y_rank == 0] = 1e-19

    # Get the lower triangle of y_rank (ignoring the diagonal).
    idx = np.where(np.tril(y_rank, k=-1))

    # Order: y1-y2, y1-y3, y2-y3, y1-y4, y2-y4, y3-y4 ... (lower triangle).
    y_rank = y_rank[idx[0], idx[1]]
    y_rank[y_rank > 0] = 1
    y_rank[y_rank <= 0] = -1

    if CUDA:
        y_rank = y_rank.cuda()

    # Make a column vector.
    y_rank = y_rank[:, np.newaxis]

    return(y_rank)


def check_predictions(all_y_hats, all_y_trus):
    """Check model predictions."""
    all_y_hats = torch.cat(all_y_hats, dim=0).cpu().detach().numpy()
    all_y_trus = torch.cat(all_y_trus, dim=0).cpu().numpy()
    total_score, pr_mu, rt_mu, rr_std, id_recall = utils.score_performance(
        all_y_hats[:, 0], all_y_trus[:, 0],
        all_y_hats[:, 1], all_y_trus[:, 1],
        all_y_hats[:, 2], all_y_trus[:, 2],
        all_y_hats[:, 3].astype(np.int32), all_y_trus[:, 3].astype(np.int32))

    return(total_score, pr_mu, rt_mu, rr_std, id_recall)


def train_loop(mdl, optimizer, loader):
    """Train model using the supplied learning rate optimizer and scheduler."""
    mdl.train(True)
    mean_loss = 0.0
    all_y_hats, all_y_trus = [], []

    for batch_idx, (X, y) in enumerate(loader):

        optimizer.zero_grad()

        if CUDA:
            X = X.cuda()
            y = y.cuda()

        y_hats = mdl.forward(X)
        losses, y_hats = calc_losses(y_hats, y, mdl.out_dims)

        # Backprop with sum of losses across prediction tasks.
        loss = sum(losses)
        loss.backward()
        mean_loss += loss.item()
        optimizer.step()
        all_y_hats.append(y_hats)
        all_y_trus.append(y)

    mean_loss /= (batch_idx+1)

    total_score, pr_mu, rt_mu, rr_std, id_recall = check_predictions(
        all_y_hats, all_y_trus)

    results = {
        'scores': {'total_score': total_score, 'pr_mu': pr_mu, 'rt_mu': rt_mu,
                   'rr_std': rr_std, 'id_recall': id_recall},
        'loss': {'mean': mean_loss}
    }

    return(results)


def evalu_loop(mdl, loader, return_preds=False):
    """Validation/Test evaluation loop."""
    mdl.eval()
    mean_loss = 0.0
    all_y_hats, all_y_trus = [], []

    for batch_idx, (X, y) in enumerate(loader):

        if CUDA:
            X = X.cuda()
            y = y.cuda()

        y_hats = mdl.forward(X)
        losses, y_hats = calc_losses(y_hats, y, mdl.out_dims)

        # Report sum of losses.
        loss = sum(losses)
        mean_loss += loss.item()
        all_y_hats.append(y_hats)
        all_y_trus.append(y)

    mean_loss /= (batch_idx+1)

    total_score, pr_mu, rt_mu, rr_std, id_recall = check_predictions(
        all_y_hats, all_y_trus)

    # If required (for evaluation), return the actual predictions made.
    if return_preds:
        results_y_hat = []

        # Loop through epochs.
        for y_hats in all_y_hats:

            # Loop through each prediction (n-element list).
            for y_hat in y_hats:
                results_y_hat.append(y_hat.cpu().detach().numpy())

        # Convert to a single numpy array.
        results_y_hat = np.vstack(results_y_hat)

    else:
        results_y_hat = None

    results = {
        'scores': {'total_score': total_score, 'pr_mu': pr_mu, 'rt_mu': rt_mu,
                   'rr_std': rr_std, 'id_recall': id_recall},
        'loss': {'mean': mean_loss},
        'preds': results_y_hat
    }

    return(results)


def train_mdl(mdl, optimizer):
    """
    Trains a submitted model using the submitted optimizer.
    """
    pp = pprint.PrettyPrinter(indent=4)
    LOGGER.info('+ Begin training with configuration:\n{}'.format(
        pp.pformat(CONFIG)))

    epochs = CONFIG['training']['epochs']

    load_args = {
        'batch_size': CONFIG['dataloader']['batch_size'],
        'num_workers': CONFIG['dataloader']['num_workers'],
        'shuffle': CONFIG['dataloader']['shuffle']}


    # Shuffles data between day1=test and day2=valid.
    data = utils.get_shuffled_data(
        test_p=CONFIG['dataloader']['test_proportion'])

    # Set up Dataloaders.
    train_data = utils.Data(precomputed=data['train'], augmentation=True)
    valid_data = utils.Data(precomputed=data['valid'], augmentation=False)

    train_load = torch.utils.data.DataLoader(train_data, **load_args)
    valid_load = torch.utils.data.DataLoader(valid_data, **load_args)

    # Move model to GPU if required.
    if CUDA:
        mdl = mdl.cuda()

    # Initial values.
    valid_loss = 10000
    best_valid_loss = 10000

    all_train_losses, all_valid_losses = [], []
    all_train_scores, all_valid_scores = [], []

    # Reduce learning rate if we plateau (valid_loss does not decrease).
    scheduler = ReduceLROnPlateau(optimizer,
        patience=CONFIG['training']['schedule_patience'])

    for ep in range(epochs):

        t1 = time.time()

        scheduler.step(valid_loss)
        train_results = train_loop(mdl, optimizer, train_load)
        valid_results = evalu_loop(mdl, valid_load)

        # Keep track of per-epoch stats for plots.
        all_train_losses.append(train_results['loss']['mean'])
        all_valid_losses.append(valid_results['loss']['mean'])
        all_train_scores.append([
            train_results['scores']['pr_mu'],
            train_results['scores']['rt_mu'],
            train_results['scores']['rr_std'],
            train_results['scores']['id_recall'],
            train_results['scores']['total_score']])
        all_valid_scores.append([
            valid_results['scores']['pr_mu'],
            valid_results['scores']['rt_mu'],
            valid_results['scores']['rr_std'],
            valid_results['scores']['id_recall'],
            valid_results['scores']['total_score']])

        # Get the best model (early stopping).
        if valid_results['loss']['mean'] < best_valid_loss:
            best_valid_loss = all_valid_losses[-1]
            best_model = mdl.state_dict()
            best_epoch = ep+1
            LOGGER.info('+ New best model found: loss={}, score={}'.format(
                best_valid_loss, valid_results['scores']['total_score']))

        # Log training performance.
        time_elapsed = time.time() - t1

        msg_info = '[{}/{}] {:.2f} sec: '.format(
            ep+1, epochs, time_elapsed)
        msg_loss = 'loss(t/v)={:.2f}/{:.2f}, '.format(
            train_results['loss']['mean'],
            valid_results['loss']['mean'])
        msg_scr1 = '{:.2f}/{:.2f}'.format(
            train_results['scores']['pr_mu'],
            valid_results['scores']['pr_mu'])
        msg_scr2 = '{:.2f}/{:.2f}'.format(
            train_results['scores']['rt_mu'],
            valid_results['scores']['rt_mu'])
        msg_scr3 = '{:.2f}/{:.2f}'.format(
            train_results['scores']['rr_std'],
            valid_results['scores']['rr_std'])
        msg_scr4 = '{:.2f}/{:.2f}'.format(
            train_results['scores']['id_recall'],
            valid_results['scores']['id_recall'])
        msg_scrt = '{:.2f}/{:.2f}'.format(
            train_results['scores']['total_score'],
            valid_results['scores']['total_score'])
        msg_task = 'scores(t/v)=[{} + {} + {} + {} = {}]'.format(
            msg_scr1, msg_scr2, msg_scr3, msg_scr4, msg_scrt)

        LOGGER.info(msg_info + msg_loss + msg_task)

        # Early stopping patience breaks training if we are just overfitting.
        if ep+1 >= best_epoch + CONFIG['training']['early_stopping_patience']:
            LOGGER.info('Impatient! No gen. improvement in {} epochs'.format(
                CONFIG['training']['early_stopping_patience']))
            break

    # Rewind to best epoch.
    LOGGER.info('Early Stopping: Rewinding to epoch {}'.format(best_epoch))
    mdl.load_state_dict(best_model)

    # Stack scores into one numpy array each.
    all_train_losses = np.vstack(all_train_losses)
    all_train_scores = np.vstack(all_train_scores)
    all_valid_losses = np.vstack(all_valid_losses)
    all_valid_scores = np.vstack(all_valid_scores)

    results = {
        'train': {'losses': all_train_losses, 'scores': all_train_scores},
        'valid': {'losses': all_valid_losses, 'scores': all_valid_scores},
        'best_epoch': best_epoch,
        'ymap': train_data.ymap
    }

    return(mdl, results)


def tspec():
    """Trains a timeseries/spectra classifier."""

    ts_len   = CONFIG['models']['tspec']['ts_len']
    spec_len = CONFIG['models']['tspec']['spec_len']
    hid_dim  = CONFIG['models']['tspec']['hid_dim']
    layers   = CONFIG['models']['tspec']['num_layers']
    out_dims = CONFIG['models']['tspec']['out_dims']
    lr       = CONFIG['training']['learning_rate']
    momentum = CONFIG['training']['momentum']
    l2       = CONFIG['training']['l2']

    mdl = models.TSpec(ts_len, spec_len, hid_dim, layers, out_dims)
    optimizer = optim.SGD(
        mdl.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

    results = train_mdl(mdl, optimizer)

    return(results)


