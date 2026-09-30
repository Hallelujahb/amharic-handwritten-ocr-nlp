# Methodology

- Same CRNN and preprocessing everywhere: grayscale, resize to 512x64, scale to 0-1
- Metrics: CER and WER from jiwer with greedy decoding, exact match also logged
- Seed 42, one run per experiment
- Baselines: AdamW, lr 1e-3, weight decay 1e-4, up to 20 epochs, patience 4
- Fine-tune: start from the AHL-D checkpoint, lr 1e-5, 10 epochs, keeps the AHL-D vocabulary
- Checkpoint chosen on validation CER
