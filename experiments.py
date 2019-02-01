from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import time
import torch
import torch.optim as optim

import models
import utils

CUDA = torch.cuda.is_available()
CONFIG = utils.read_config()


def calc_losses(y_hats, y_train, out_types, reg_loss, clf_loss, verb=False):
    """Calculate all losses across all prediction tasks."""

    losses = []
    scores = []

    msg = 'scores: '

    for i in range(len(out_types)):
        y_hat = y_hats[i]
        y_tru = y_train[:, i]

        if out_types[i] == 'regression':
            loss = reg_loss(y_hat, y_tru)
            losses.append(loss)

            # Mean squared error.
            scores.append(loss)
            msg += 'mse={:.2f} '.format(loss.item())

        elif out_types[i] == 'classification':
            loss = clf_loss(y_hat, y_tru.long())
            losses.append(loss)

            # Accuracy.
            _, preds = torch.max(y_hat.data, 1)
            n_correct = torch.sum(preds == y_tru.long().data)
            scores.append(n_correct)
            msg += 'acc={:.2f} '.format(n_correct.item())

    if verb:
        print(msg)

    return(losses, scores)


def train(mdl, optimizer):

    loader_params = {
        'batch_size': CONFIG['dataloader']['batch_size'],
        'shuffle': CONFIG['dataloader']['shuffle'],
        'num_workers': CONFIG['dataloader']['num_workers']
    }

    out_dims = CONFIG['models']['lstm']['out_dims']
    out_types = CONFIG['models']['lstm']['out_types']

    reg_loss = MSELoss()
    clf_loss = CrossEntropyLoss()

    # Reduce learning rate if we plateau (valid_loss does not decrease)
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    valid_loss = 10000 # initial value

    # Set up Dataloaders.
    train_data = utils.Data(augmentation=True)
    valid_data = utils.Data(train=False, augmentation=True)
    train_load = data.DataLoader(train_data, **loader_params)
    valid_load = data.DataLoader(valid_data, **loader_params)

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

        for batch_idx, (X_train, y_train) in enumerate(train_load):

            optimizer.zero_grad()

            if CUDA:
                X_train = X_train.cuda()
                y_train = y_train.cuda()

            # All lengths are 1 for this experiment (only one timeseries).
            y_hats = mdl.forward(X_train.unsqueeze(1))

            # Backprop with sum of losses across prediction tasks.
            losses, scores= calc_losses(
                y_hats, y_train, out_types, reg_loss, clf_loss
            )

            loss = sum(losses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (batch_idx+1)

        # Validation loop.
        mdl.eval()
        valid_loss = 0.0

        for batch_idx, (X_valid, y_valid) in enumerate(valid_load):

            # Transfer to GPU
            if CUDA:
                X_valid = X_valid.cuda()
                y_valid = y_valid.cuda()

            # Model computations
            y_hats = mdl.forward(X_valid.unsqueeze(1))

            losses, scores = calc_losses(
                y_hats, y_valid, out_types, reg_loss, clf_loss, verb=True
            )

            loss = sum(losses)
            valid_loss += loss.item()

        valid_loss /= (batch_idx+1)

        t2 = time.time()
        time_elapsed = t2-t1

        print('[{}/{}] {:.2f} sec: loss(train/valid)={:.4f}/{:.4f}'.format(
            ep+1, CONFIG['training']['epochs'],
            time_elapsed, train_loss, valid_loss)
        )


def lstm():
    """Trains a LSTM classifier."""

    seq_len  = CONFIG['models']['lstm']['sequence_len']
    hid_dim  = CONFIG['models']['lstm']['hidden_dim']
    layers   = CONFIG['models']['lstm']['num_layers']
    dropout  = CONFIG['models']['lstm']['dropout']
    out_dims = CONFIG['models']['lstm']['out_dims']
    lr       = CONFIG['training']['learning_rate']
    momentum = CONFIG['training']['momentum']
    l2       = CONFIG['training']['l2']

    mdl = models.LSTMClassifier(seq_len, hid_dim, layers, dropout, out_dims)
    optimizer = optim.SGD(
        mdl.parameters(), lr=lr, momentum=momentum, weight_decay=l2
    )

    results = train(mdl, optimizer)

    return(results)


