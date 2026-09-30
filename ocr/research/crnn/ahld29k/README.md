# AHL-D 29K CRNN baseline

Training: 23,911 train / 3,046 val / 2,990 test, 144 / 18 / 18 writers. AdamW lr 1e-3, weight decay 1e-4, ReduceLROnPlateau on CER, 20 epochs.

Best epoch 20: val CER 4.62%, WER 18.07%, exact match 30.1%.

`val_predictions.csv` is the validation set. `first_run/` has the config files of the earlier attempt (21,326 / 2,585 split, cpu), kept for reference.
