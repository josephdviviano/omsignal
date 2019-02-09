OMsignal
--------

Here we propose a simple model that operates on ECG data using 1d convolutions on the time domain, and simultaneously analyzes the spectra of these timeseries using a some fully-connected layers. The model employs batch normalization throughout, with ReLU activations. This model is trained to jointly predict 3 real numbers (mean r-r interval, std r-r interval, mean t-r interval) and the participant ID (32 classes).

`main.py` can be executed to run all experiments. It will train a model using `config.yml` and output all results in `logs/`, `models/`, and `img`/.

`evaluation/eval.py` can be executed to run the best trained model against the test set.

All major training settings are handled in `config.yml`. Most options should be self-explainatory. The remaining ones are handled below:

**config.yml**:

+ `data`: location of the input files.
+ `preprocessing`:
    + `noise_gain`: scaling factor applied to additive noise for data augmentation.
+ `dataloader`:
    + `test_proportion`: we combine and shuffle the training and validation sets to increase our training set size. This number [0 1] controls the proportion allocated to the validation set.
+ `models`:
    + `tspec`:
        + `ts_len`: length of the input timeseries in X.
        + `spec_len`: length of the spectra of the input timeseries in X (typically ts_len/2).
        + `hid_dim`: size of all hidden layers in the network. Controls capacity.
        + `num_layers`: number of hidden layers of the spectra MLP component and the shared embedding components of the network. Controls capacity.
        + `out_dims`: a list of the predictor dimensions. If an element is 1, the task is treated as a regression. Otherwise it is treated as an n-class classification problem.
+ `training`:
    + `schedule_patience`: number of epochs where the validation loss does not decrease to wait before decreasing the learning rate by an order of magnitude.
    + `early_stopping_patience`: number of epochs to wait where the validation loss does not decrease before deciding to stop training altogether.

**Training Details**:

This model is trained on all 4 tasks jointly. For evaluation, kendal's tau is used (a rank-order correlation measure) for all regression tasks. Therefore, `MarginRankLoss` is employed instead of `MSELoss`, as would be typically used for regression tasks. Classification is trained using `CrossEntropyLoss`. Optimization was done using stochastic gradient descent with momentum. A learning rate scheduler is employed such that the learning rate is reduced by an order of magnitude if the validation loss plateaus for more than 20 epochs.

**Data:**

The provided data is split into 3 binary files:

+ `MILA_TrainLabeledData.dat` - labeled data for supervised training
+ `MILA_ValidationLabeledData.dat` -  labeled data for validation,
+ `MILA_UnlabeledData.dat` -  unlabeled data.

Each labeled dataset (`MILA_TrainLabeledData.dat` and `MILA_ValidationLabeledData.dat`) contains 5 windows of 30 second length ECG data (sampled at 125 Hz) for each of the 32 participants. For each participant, the samples in the `MILA_TrainLabeledData.dat` and `MILA_ValidationLabeledData.dat` datasets have were collected on independent days. The test data was collected on a 3rd day.

The labeled data looks like this:

+ `Shape = 160 x 3754` - where `160 = 5 x 32 ` corresponds to the number of windows.
+ `Column 0` to `Column 3749` - Columns corresponding to the ECG data ( `30 seconds x 125 Hz = 3750` ). They contain `float` values.
+ `Column 3750` - Columns corresponding to the `PR_Mean` of the corresponding ECG sample. It contains `float` values.
+ `Column 3751` - Columns corresponding to the `RT_Mean` of the corresponding ECG sample. It contains `float` values.
+ `Column 3752` - Columns corresponding to the `RR_StdDev` of the corresponding ECG sample. It contains `float` values.
+ `Column 3753` - Columns corresponding to the `ID` of the participant. It contains `int` values.

The unlabeled data looks like this:

+ `Shape = 657233 x 3750` - where `657233 ` corresponds to the remaining number of unlabeled windows.
+ `Column 0` to `Column 3749` - Columns corresponding to the ECG data ( `30 seconds x 125 Hz = 3750` ). They contain `float` values.

The `ID` column of the dataset can be mapped back and forth between the original range `[0 43]` and a machine-friendly range `[0 32]` using the `ymap()` method of the Data object: `utils.Data.ymap()`. This is used for reporting final values during evaluation.

