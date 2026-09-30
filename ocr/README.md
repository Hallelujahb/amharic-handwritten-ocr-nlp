# OCR

- `notebooks/` run the experiments in Colab
- `src/` reusable code, run from inside `ocr/src`
- `research/crnn/` is the only experiment with results so far

Transcribe line images with a checkpoint:

    cd ocr/src
    python inference.py ../../checkpoints/ahld29k_crnn_best.pt line1.png line2.png
