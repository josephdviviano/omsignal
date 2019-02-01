#!/usr/bin/env python

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import torch
from torch.utils import data
import utils
import models

CUDA = torch.cuda.is_available()
CONFIG = utils.read_config()

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


# Hyper-parameters
sequence_length = 3749
input_size = 7499
hidden_size = 100
num_layers = 2
num_classes = 4
batch_size = 64
num_epochs = 50
learning_rate = 0.01

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(1), self.hidden_size).to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        out = self.fc(out[:,-1,:])
        out = self.softmax(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Train the model
total_step = len(train_data)
for epoch in range(num_epochs):
    # Training loop
    for i,(X_train, y_train) in enumerate(train_load):

        # Transfer to GPU
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_train=torch.tensor(y_train,dtype=torch.long,device=device)
        outputs = model(X_train.unsqueeze(1)) # add num_features dimension (2)
        import IPython; IPython.embed()
        loss = criterion(outputs, y_train[:, 0])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 3 == 0:
            print("first loop")
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


    # Validation evaluation
'''
    with torch.set_grad_enabled(False):
        for X_valid, y_valid in valid_load:

            # Transfer to GPU
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)

            outputs = model(X_valid)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == y_valid).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
'''
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')







