# Amharic handwritten OCR and NLP correction

Recognizing handwritten Amharic text lines with a CRNN, then correcting the output with NLP (not started).

## Status

Only the CRNN work is done. The other model folders under `ocr/research` are empty placeholders.

| Train | Test | CER | WER |
| --- | --- | ---: | ---: |
| AHL-D | AHL-D validation | 4.62% | 18.07% |
| Fidel | Fidel validation | 16.47% | 41.07% |
| AHL-D | Fidel (all handwritten) | 49.38% | 85.71% |
| Fidel | AHL-D (all) | 41.18% | 75.46% |
| AHL-D, fine-tuned on Fidel | Fidel test | 27.97% | 66.98% |
| AHL-D, fine-tuned on Fidel | AHL-D test | 23.69% | 44.57% |

Models trained on one dataset transfer badly to the other. Fine-tuning helps on Fidel but the AHL-D score drops a lot. Full write-up in `ocr/research/crnn/README.md`.

## Layout

- `ocr/notebooks`: Colab notebooks, one per stage
- `ocr/src`: model, dataset, decoding and inference code
- `ocr/research`: results per experiment
- `ocr/data`: splits, vocabularies and configs
- `results/comparison.csv`: all numbers in one table
- `checkpoints/`: model weights, not tracked by git

## Data

- AHL-D 29K: `misiker/AHLD-29k` on Hugging Face
- Fidel handwritten subset: `upanzi/fidel-dataset`

## License

MIT, see `LICENSE`.
