import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils

CUDA = torch.cuda.is_available()

class LSTMClassifier(nn.Module):

    def __init__(self, seq_len, emb_dim, hid_dim, out_dims):
        """
        output_sizes is a n-element list of sizes, one for each prediction task.
        """

        super(LSTMClassifier, self).__init__()

        # Check inputs.
        if len(out_dims) != 4:
            raise ValueError('out_dims should have length 4.')

        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.out_dims = out_dims

        # LSTM accepts the inputs.
        self.lstm = nn.LSTM(seq_len, hid_dim, num_layers=1, batch_first=True)

        # Dropout applied to the last hidden state of the LSTM.
        self.dropout_layer = nn.Dropout(p=0.2)

        self.out1 = nn.Linear(hid_dim, out_dims[0])
        self.out2 = nn.Linear(hid_dim, out_dims[1])
        self.out3 = nn.Linear(hid_dim, out_dims[2])
        self.out4 = nn.Linear(hid_dim, out_dims[3])

    def _init_hidden(self, batch_size):
        ht = autograd.Variable(torch.randn(1, batch_size, self.hid_dim))
        ct = autograd.Variable(torch.randn(1, batch_size, self.hid_dim))

        if CUDA:
            ht = ht.cuda()
            ct = ct.cuda()

        return((ht,ct))


    def forward(self, X):
        """
        X is size=(batch_size, n_timepoints)
        lengths is the number of paired sequences in X (typically 1).
        """

        # Initialize hidden states of LSTM.
        self.hidden = self._init_hidden(X.size(0))

        _, (ht, ct) = self.lstm(X, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        y_hat = self.dropout_layer(ht[-1])

        # All output heads operate on the same LSTM state.
        y_hat_1 = self.out1(y_hat)
        y_hat_2 = self.out2(y_hat)
        y_hat_3 = self.out3(y_hat)
        y_hat_4 = self.out4(y_hat)

        return([y_hat_1, y_hat_2, y_hat_3, y_hat_4])


