from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import logging
import numpy as np
import time
import torch
import torch.optim as optim
import os

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
            losses.append(reg_loss(y_hat, y_tru))
            predictions.append(y_hat)

        # Classification case.
        elif out_dim > 1:
            losses.append(clf_loss(y_hat, y_tru.long()))
            _, preds = torch.max(y_hat.data, 1)
            predictions.append(preds.float().unsqueeze(1))
            #print('pred: {}'.format(preds))
            #print('true: {}'.format(y_tru))

    predictions = torch.cat(predictions, dim=1)

    return(losses, predictions)


def train(mdl, optimizer):

    out_dims = CONFIG['models']['lstm']['out_dims']
    loss_weights = CONFIG['training']['loss_weights']

    reg_loss = MSELoss()
    clf_loss = CrossEntropyLoss()

    # Reduce learning rate if we plateau (valid_loss does not decrease)
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    valid_loss = 10000 # initial value

    # Shuffles data between day1=test and day2=valid.
    data = utils.get_shuffled_data(
        test_p=CONFIG['dataloader']['test_proportion']
    )
    train_data = utils.Data(precomputed=data['train'], augmentation=True)
    valid_data = utils.Data(precomputed=data['valid'], augmentation=False)

    # Set up Dataloaders.
    train_load = data.DataLoader(train_data,
        batch_size=CONFIG['dataloader']['batch_size'],
        num_workers=CONFIG['dataloader']['num_workers'],
        shuffle=CONFIG['dataloader']['shuffle']
    )

    # Can't shuffle the valid_load so we can use evaluation script.
    valid_load = data.DataLoader(valid_data,
        batch_size=CONFIG['dataloader']['batch_size'],
        num_workers=CONFIG['dataloader']['num_workers']
    )

    # Configure CUDA.
    #device = torch.device("cuda:0" if CUDA else "cpu")
    #torch.backends.cudnn.benchmark = True

    # Move model to GPU if required.
    if CUDA:
        mdl = mdl.cuda()
        reg_loss = reg_loss.cuda()
        clf_loss = clf_loss.cuda()

    for ep in range(CONFIG['training']['epochs']):

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

            # Scale losses into the same range (config.yml).
            normalized_losses = []
            for i, loss in enumerate(losses):
                normalized_losses.append(loss * loss_weights[i])

            # Backprop with sum of losses across prediction tasks.
            loss = sum(normalized_losses)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            all_y_hats.append(y_hats)
            all_y_trus.append(y_train)

        train_loss /= (batch_idx+1)

        # aggregate predictions and check them here
        all_y_hats = torch.cat(all_y_hats, dim=0).cpu().detach().numpy()
        all_y_trus = torch.cat(all_y_trus, dim=0).cpu().numpy()
        t_scrt, t_scr1, t_scr2, t_scr3, t_scr4 = utils.scorePerformance(
            all_y_hats[:, 0], all_y_trus[:, 0],
            all_y_hats[:, 1], all_y_trus[:, 1],
            all_y_hats[:, 2], all_y_trus[:, 2],
            all_y_hats[:, 3].astype(np.int32), all_y_trus[:, 3].astype(np.int32)
        )

        # Validation loop.
        mdl.eval()
        valid_loss = 0.0
        all_y_hats, all_y_trus = [], []

        for batch_idx, (X_valid, y_valid) in enumerate(valid_load):

            # Transfer to GPU
            if CUDA:
                X_valid = X_valid.cuda()
                y_valid = y_valid.cuda()

            y_hats = mdl.forward(X_valid)
            losses, y_hats = calc_losses(
                y_hats, y_valid, out_dims, reg_loss, clf_loss)

            # Scale losses to the same range.
            normalized_losses = []
            for i, loss in enumerate(losses):
                normalized_losses.append(loss * loss_weights[i])

            loss = sum(normalized_losses)
            valid_loss += loss.item()
            all_y_hats.append(y_hats)
            all_y_trus.append(y_valid)

        valid_loss /= (batch_idx+1)

        # aggregate predictions and check them here
        all_y_hats = torch.cat(all_y_hats, dim=0).cpu().detach().numpy()
        all_y_trus = torch.cat(all_y_trus, dim=0).cpu().numpy()
        v_scrt, v_scr1, v_scr2, v_scr3, v_scr4 = utils.scorePerformance(
            all_y_hats[:, 0], all_y_trus[:, 0],
            all_y_hats[:, 1], all_y_trus[:, 1],
            all_y_hats[:, 2], all_y_trus[:, 2],
            all_y_hats[:, 3].astype(np.int32), all_y_trus[:, 3].astype(np.int32)
        )

        # Log training performance.
        time_elapsed = time.time() - t1

        msg_info = '[{}/{}] {:.2f} sec: '.format(
            ep+1, CONFIG['training']['epochs'], time_elapsed
        )
        msg_loss = 'loss(t/v)={:.2f}/{:.2f}, '.format(train_loss, valid_loss)
        msg_scr1 = '{:.2f}/{:.2f}'.format(t_scr1, v_scr1)
        msg_scr2 = '{:.2f}/{:.2f}'.format(t_scr2, v_scr2)
        msg_scr3 = '{:.2f}/{:.2f}'.format(t_scr3, v_scr3)
        msg_scr4 = '{:.2f}/{:.2f}'.format(t_scr4, v_scr4)
        msg_scrt = '{:.2f}/{:.2f}'.format(t_scrt, v_scrt)
        msg_task = 'scores(t/v)=[{} + {} + {} + {} = {}]'.format(
            msg_scr1, msg_scr2, msg_scr3, msg_scr4, msg_scrt
        )

        LOGGER.info(msg_info + msg_loss + msg_task)


def lstm():
    """Trains a LSTM classifier."""

    ts_len   = CONFIG['models']['lstm']['ts_len']
    spec_len = CONFIG['models']['lstm']['spec_len']
    hid_dim  = CONFIG['models']['lstm']['hid_dim']
    layers   = CONFIG['models']['lstm']['num_layers']
    out_dims = CONFIG['models']['lstm']['out_dims']
    lr       = CONFIG['training']['learning_rate']
    momentum = CONFIG['training']['momentum']
    l2       = CONFIG['training']['l2']

    mdl = models.ConvLSTM(ts_len, spec_len, hid_dim, layers, out_dims)
    optimizer = optim.SGD(
        mdl.parameters(), lr=lr, momentum=momentum, weight_decay=l2
    )

    results = train(mdl, optimizer)

    return(results)


