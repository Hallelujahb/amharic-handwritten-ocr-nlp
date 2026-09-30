# Fidel CRNN baseline

Train and test on Fidel handwritten, writer-disjoint split (31,569 train / 3,251 val, 372 / 41 writers). AdamW lr 1e-3, 512x64 input, batch 16, up to 20 epochs with patience 4.

Stopped at epoch 17. Best epoch 13: val CER 16.47%, WER 41.07%, exact match 1.3%.
