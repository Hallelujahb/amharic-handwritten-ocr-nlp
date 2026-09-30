# Fine-tune AHL-D to Fidel

Start: AHL-D best checkpoint. Data: Fidel handwritten, split by writer 80/10/10 with seed 42 (27,855 / 3,458 / 3,507). 10 epochs, lr 1e-5, weight decay 1e-5, patience 3.

Best epoch 10, Fidel val CER 24.72%.

Final test: Fidel test CER 27.97%, WER 66.98%. AHL-D test CER 23.69%, WER 44.57%. See `metrics.json`.
