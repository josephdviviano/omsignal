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

def calc_losses(y_hats, y, out_dims, reg_loss, clf_loss, verb=False):
    """
    Calculate all losses across all prediction tasks.
    Also reformats 'predictions' to be a friendly pytorch tensor for later use.
    """
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
    y_rank = y[np.newaxis, :] - y[:, np.newaxis]

    # Edge case where the difference between 2 points is 0.
    y_rank[y_rank == 0] = 1e-19

    idx = np.where(np.tril(y_rank, k=-1))

    # Order: y1-y2, y1-y3, y2-y3, y1-y4, y2-y4, y3-y4 ... (lower triangle).
    y_rank = y_rank[idx[0], idx[1]]
    y_rank[y_rank > 0] = 1
    y_rank[y_rank <= 0] = -1

    if CUDA:
        y_rank = y_rank.cuda()

    y_rank = y_rank[:, np.newaxis]

    return(y_rank)


def train(mdl, optimizer):
    """
    Trains a submitted model using the submitted optimizer.
    """
    pp = pprint.PrettyPrinter(indent=4)

    LOGGER.info('--- Begin Training with configuration:\n{}'.format(
        pp.pformat(CONFIG)))

    out_dims = CONFIG['models']['tspec']['out_dims']
    epochs = CONFIG['training']['epochs']

    reg_loss = MarginRankingLoss()
    clf_loss = CrossEntropyLoss()

    # Reduce learning rate if we plateau (valid_loss does not decrease)
    scheduler = ReduceLROnPlateau(optimizer, patience=20)
    valid_loss = 10000 # initial value

    # Shuffles data between day1=test and day2=valid.
    data = utils.get_shuffled_data(
        test_p=CONFIG['dataloader']['test_proportion'])

    train_data = utils.Data(precomputed=data['train'], augmentation=True)
    valid_data = utils.Data(precomputed=data['valid'], augmentation=False)

    load_args = {
        'batch_size': CONFIG['dataloader']['batch_size'],
        'num_workers': CONFIG['dataloader']['num_workers'],
        'shuffle': CONFIG['dataloader']['shuffle']}

    # Set up Dataloaders.
    train_load = torch.utils.data.DataLoader(train_data, **load_args)
    valid_load = torch.utils.data.DataLoader(valid_data, **load_args)

    # Move model to GPU if required.
    if CUDA:
        mdl = mdl.cuda()
        reg_loss = reg_loss.cuda()
        clf_loss = clf_loss.cuda()

    for ep in range(epochs):

        t1 = time.time()

        # Train loop.
        scheduler.step(valid_loss)
        mdl.train(True)
        train_loss = 0.0
        all_y_hats, all_y_trus = [], []

        for batch_idx, (X_train, y_train) in enumerate(train_load):

            optimizer.zero_grad()

            if CUDA:
                X_train = X_train.cuda()
                y_train = y_train.cuda()

            y_hats = mdl.forward(X_train)
            losses, y_hats = calc_losses(
                y_hats, y_train, out_dims, reg_loss, clf_loss)

            # Backprop with sum of losses across prediction tasks.
            loss = sum(losses)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            all_y_hats.append(y_hats)
            all_y_trus.append(y_train)

        train_loss /= (batch_idx+1)

        # Check predictions.
        all_y_hats = torch.cat(all_y_hats, dim=0).cpu().detach().numpy()
        all_y_trus = torch.cat(all_y_trus, dim=0).cpu().numpy()
        t_scrt, t_scr1, t_scr2, t_scr3, t_scr4 = utils.score_performance(
            all_y_hats[:, 0], all_y_trus[:, 0],
            all_y_hats[:, 1], all_y_trus[:, 1],
            all_y_hats[:, 2], all_y_trus[:, 2],
            all_y_hats[:, 3].astype(np.int32),
            all_y_trus[:, 3].astype(np.int32))

        # Validation loop.
        mdl.eval()
        valid_loss = 0.0
        all_y_hats, all_y_trus = [], []

        for batch_idx, (X_valid, y_valid) in enumerate(valid_load):

            if CUDA:
                X_valid = X_valid.cuda()
                y_valid = y_valid.cuda()

            y_hats = mdl.forward(X_valid)
            losses, y_hats = calc_losses(
                y_hats, y_valid, out_dims, reg_loss, clf_loss)

            # Report sum of losses.
            loss = sum(losses)
            valid_loss += loss.item()
            all_y_hats.append(y_hats)
            all_y_trus.append(y_valid)

        valid_loss /= (batch_idx+1)

        # Check predictions.
        all_y_hats = torch.cat(all_y_hats, dim=0).cpu().detach().numpy()
        all_y_trus = torch.cat(all_y_trus, dim=0).cpu().numpy()
        v_scrt, v_scr1, v_scr2, v_scr3, v_scr4 = utils.score_performance(
            all_y_hats[:, 0], all_y_trus[:, 0],
            all_y_hats[:, 1], all_y_trus[:, 1],
            all_y_hats[:, 2], all_y_trus[:, 2],
            all_y_hats[:, 3].astype(np.int32),
            all_y_trus[:, 3].astype(np.int32))

        # Log training performance.
        time_elapsed = time.time() - t1

        msg_info = '[{}/{}] {:.2f} sec: '.format(
            ep+1, epochs, time_elapsed)
        msg_loss = 'loss(t/v)={:.2f}/{:.2f}, '.format(train_loss, valid_loss)
        msg_scr1 = '{:.2f}/{:.2f}'.format(t_scr1, v_scr1)
        msg_scr2 = '{:.2f}/{:.2f}'.format(t_scr2, v_scr2)
        msg_scr3 = '{:.2f}/{:.2f}'.format(t_scr3, v_scr3)
        msg_scr4 = '{:.2f}/{:.2f}'.format(t_scr4, v_scr4)
        msg_scrt = '{:.2f}/{:.2f}'.format(t_scrt, v_scrt)
        msg_task = 'scores(t/v)=[{} + {} + {} + {} = {}]'.format(
            msg_scr1, msg_scr2, msg_scr3, msg_scr4, msg_scrt)

        LOGGER.info(msg_info + msg_loss + msg_task)


def tspec():
    """Trains a timeseries/spectra classifier."""

    ts_len   = CONFIG['models']['tspec']['ts_len']
    spec_len = CONFIG['models']['tspec']['spec_len']
    hid_dim  = CONFIG['models']['tspec']['hid_dim']
    layers   = CONFIG['models']['tspec']['num_layers']
    out_dims = CONFIG['models']['tspec']['out_dims']
    freeze   = CONFIG['models']['tspec']['freeze']
    lr       = CONFIG['training']['learning_rate']
    momentum = CONFIG['training']['momentum']
    l2       = CONFIG['training']['l2']

    mdl = models.TSpec(ts_len, spec_len, hid_dim, layers, out_dims, freeze)
    optimizer = optim.SGD(
        mdl.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

    results = train(mdl, optimizer)

    return(results)


