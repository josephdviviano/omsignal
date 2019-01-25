import torch
from torch.utils import data
import utils
import models

CUDA = torch.cuda.is_available()
CONFIG = utils.read_config()


def train(mdl):

    params = {
        'batch_size': CONFIG['dataloader']['batch_size'],
        'shuffle': CONFIG['dataloader']['shuffle'],
        'num_workers': CONFIG['dataloader']['num_workers']
    }

    # set up Dataloaders
    train_data = utils.Data(augmentation=True)
    valid_data = utils.Data(train=False, augmentation=True)
    train_load = data.DataLoader(train_data, **params)
    valid_load = data.DataLoader(valid_data, **params)

    # configure cuda
    device = torch.device("cuda:0" if CUDA else "cpu")
    torch.backends.cudnn.benchmark = True

    for epoch in range(CONFIG['training']['epochs']):

        # Training loop
        for X_train, y_train in train_load:

            # Transfer to GPU
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Model computations
            #mdl.forward()

        # Validation evaluation
        with torch.set_grad_enabled(False):
            for X_valid, y_valid in valid_load:

                # Transfer to GPU
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)

                # Model computations
                #predictions = mdl.forward(X_valid)


def lstm():
    """Trains a LSTM classifier."""
    mdl = models.LSTMClassifier()
    results = train(mdl)

    return(results)


