import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils

CUDA = torch.cuda.is_available()

class LSTMClassifier(nn.Module):

    def __init__(self, ts_len, spec_len, hid_dim, layers, dropout, out_dims):
        """
        output_sizes is a n-element list of sizes, one for each prediction task.
        """
        super(LSTMClassifier, self).__init__()

        # Check inputs.
        if len(out_dims) != 4:
            raise ValueError('out_dims should have length 4.')

        self.ts_len = ts_len
        self.spec_len = spec_len

        self.hid_dim = hid_dim
        self.layers = layers
        self.out_dims = out_dims

        self.initalized = False

        # LSTM accepts the timeseries input.
        self.lstm = nn.LSTM(
            ts_len, hid_dim, num_layers=layers, batch_first=True
        )

        # MLP accepts the spectra. Linear-->Relu-->Dropout
        # No dropout on final layer.
        arch = []
        for i in range(layers):

            if i == 0:
                arch.append(nn.Linear(spec_len, hid_dim))
            else:
                arch.append(nn.Linear(hid_dim, hid_dim))

            arch.append(nn.ReLU())

            if i != layers-1:
                arch.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*arch)

        # Dropout applied to the last hidden state of the LSTM concatenated
        # with the outputs of the MLP.
        self.dropout_layer = nn.Dropout(p=dropout)

        # Output heads are hard-coded to have 3 fully connected layers.
        # TODO: this should be a setting in config and made elegant.
        self.out1 = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dims[0])
        )
        self.out2 = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dims[1])
        )
        self.out3 = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dims[2])
        )
        self.out4 = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dims[3])
        )

    def _init_hidden(self, bs):
        ht = autograd.Variable(torch.randn(self.layers, bs, self.hid_dim))
        ct = autograd.Variable(torch.randn(self.layers, bs, self.hid_dim))

        if CUDA:
            ht = ht.cuda()
            ct = ct.cuda()

        return((ht,ct))

    def _initalize(self, init_type='glorot'):
        """
        model     -- a pytorch sequential model
        init_type -- one of 'zero', 'normal', 'glorot'
        Takes in a model, initializes it to all-zero, normal distribution
        sampled, or glorot initialization. Golorot == xavier.
        """
        if init_type not in ['zero', 'normal', 'glorot']:
            raise Exception('init_type invalid]')

        for k, v in self.mlp.named_parameters():
            if k.endswith('weight'):
                if init_type == 'zero':
                    torch.nn.init.constant(v, 0)
                elif init_type == 'normal':
                    torch.nn.init.normal(v)
                elif init_type == 'glorot':
                    torch.nn.init.xavier_uniform(v, gain=calculate_gain('relu'))
                else:
                    raise Exception('invalid init_type')

    def forward(self, X):
        """
        X is size=(batch_size, ts_len+spec_len).
        We use self.ts_len and self.spec_len to split X to be fed into
        the LSTM head and MLP head.
        """
        if not self.initalized:
            self._initalize

        batch_size = X.size(0)

        X_ts = X[:, :, :self.ts_len]
        X_spec = X[:, :, self.ts_len:]

        # Initialize hidden states of LSTM.
        self.hidden = self._init_hidden(batch_size)

        # Pass timeseries through LSTM.
        _, (ht, ct) = self.lstm(X_ts, self.hidden)

        # Pass spectra through MLP.
        mlp_activations = self.mlp(X_spec)

        # Hidden state is the concatenation of both.
        # ht is the last hidden state of the sequences.
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        hid = torch.cat([ht[-1], mlp_activations.squeeze(1)], dim=1)

        # Dropout on concatenated hidden state.
        y_hat = self.dropout_layer(hid)

        # All output heads operate on the same LSTM state.
        # TODO: Use nn.modulelist here for arbitrary number of outputs.
        y_hat_1 = self.out1(y_hat)
        y_hat_2 = self.out2(y_hat)
        y_hat_3 = self.out3(y_hat)
        y_hat_4 = self.out4(y_hat)

        return([y_hat_1, y_hat_2, y_hat_3, y_hat_4])


