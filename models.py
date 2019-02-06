import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

import utils

CUDA = torch.cuda.is_available()

class TSpec(nn.Module):

    def __init__(self, ts_len, spec_len, hid_dim, layers, out_dims):
        """
        Model that accepts, as input, a timeseries concatenated with the
        spectra of that timeseries. The timeseries is fed through a 1DCNN
        to extract interesting shapes in the signal. Simultaneously, the
        spectra of the timeseries is analyzed by a seperate MLP head, which
        learns about informative peaks in the spectra. The outputs of these
        two paths are then concatenated and fed through an embedding MLP.
        Finally, for the n outputs requested, single MLP layer is used to
        predict either a real number (regression) or distribution
        (classification).

        ts_len:   Number of timepoints in the timeseries (CNN->LSTM path).
        spec_len: Number of frequency bins in the spectra (MLP).
        hid_dim:  Controls the size of all intermediate layers.
        layers:   Number of layers for the CNN, MLP, and embedding components.
        out_dims: List of integers for the size of each output head. One
                  for each prediction task. Regression == 1,
                  Classification >= 1.

        """
        super(TSpec, self).__init__()

        self.ts_len = ts_len
        self.spec_len = spec_len
        self.hid_dim = hid_dim
        self.layers = layers
        self.out_dims = out_dims

        # 5-layer CNN accepts the timeseries input.
        # Use mean-pooling so we are more sensitive to exact mean R-R times.
        # Conv --> AvgPool --> BatchNorm --> ReLU.
        self.conv = nn.Sequential(
            nn.Conv1d(1, hid_dim, 5),
            nn.AvgPool1d(5),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Conv1d(hid_dim, hid_dim, 5),
            nn.AvgPool1d(5),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Conv1d(hid_dim, hid_dim, 5),
            nn.AvgPool1d(5),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Conv1d(hid_dim, hid_dim, 5),
            nn.AvgPool1d(5),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Conv1d(hid_dim, hid_dim, 3),
            nn.AvgPool1d(2),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        # n-layer MLP accepts the spectra. Linear --> Batchnorm --> ReLU
        # Minimum 2-layers, first layer always embeds to FIXED neurons.
        FIXED = 1000
        arch = []
        arch.append(nn.Linear(spec_len, FIXED))
        arch.append(nn.BatchNorm1d(FIXED))
        arch.append(nn.ReLU())
        for i in range(layers):
            if i == 0:
                arch.append(nn.Linear(FIXED, hid_dim))
            else:
                arch.append(nn.Linear(hid_dim, hid_dim))
            arch.append(nn.BatchNorm1d(hid_dim))
            arch.append(nn.ReLU())
        self.mlp = nn.Sequential(*arch)

        # Embedding mixes the timeseries and spectral representations.
        # Linear --> BatchNorm --> ReLU.
        arch = []
        for i in range(layers):
            if i == 0:
                arch.append(nn.Linear(hid_dim*2, hid_dim))
            else:
                arch.append(nn.Linear(hid_dim, hid_dim))
            arch.append(nn.BatchNorm1d(hid_dim))
            arch.append(nn.ReLU())
        self.embedding = nn.Sequential(*arch)

        # Output heads are a single fully connected layer.
        self.outputs = nn.ModuleList([])
        for out_dim in out_dims:
            self.outputs.append(nn.Linear(hid_dim, out_dim))

    def _init_hidden(self, bs):
        ht = autograd.Variable(torch.randn(self.layers, bs, self.hid_dim))
        ct = autograd.Variable(torch.randn(self.layers, bs, self.hid_dim))

        if CUDA:
            ht = ht.cuda()
            ct = ct.cuda()

        return((ht,ct))

    def forward(self, X):
        """
        X is size=(batch_size, ts_len+spec_len).
        We use self.ts_len and self.spec_len to split X to be fed into
        the CNN head and MLP head.
        """
        batch_size = X.size(0)

        X_time = X[:, :self.ts_len]
        X_spec = X[:, self.ts_len:]

        # Convolutional step on timeseries.
        conv_act = self.conv(X_time.unsqueeze(1))

        # Initialize hidden states of LSTM.
        #self.hidden = self._init_hidden(batch_size)

        # Pass convolutional activations through LSTM.
        #_, (ht, ct) = self.lstm(conv_act.transpose(1, 2), self.hidden)

        # ht is the last hidden state of the sequences.
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        #lstm_act = ht[-1]

        # Pass spectra through MLP.
        mlp_act = self.mlp(X_spec)

        # Hidden state is the concatenation lstm and mlp branches.
        #hid = torch.cat([lstm_act, mlp_act], dim=1)
        hid = torch.cat([conv_act.squeeze(), mlp_act], dim=1)

        # Embed mixed representations from CNN and MLP.
        y_hat = self.embedding(hid)

        # Generate individual predictions from this embedding.
        y_hats = []
        for i, output in enumerate(self.outputs):
            y_hats.append(output(y_hat))

        return(y_hats)

