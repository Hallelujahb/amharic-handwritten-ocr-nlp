# CRNN

Everything run with the CRNN so far, in the order it happened. Notebooks are in `ocr/notebooks`, combined numbers in `results/comparison.csv`.

Datasets: AHL-D 29K (`misiker/AHLD-29k`) and the handwritten subset of Fidel (`upanzi/fidel-dataset`). Input 512x64 grayscale, CTC loss, AdamW, greedy decoding.

## 1. AHL-D baseline (`ahld29k/`)

Trained on AHL-D with the HF train split, best epoch 20. On the 3,046 sample validation split: CER 4.62%, WER 18.07%.
This is the validation split that was used to pick the checkpoint. The 2,990 sample test split has not been scored for the baseline yet (notebook 06).

## 2. Fidel baseline (`extended/`)

Trained on Fidel handwritten (34,820 samples, 31,569 train / 3,251 val, writer-disjoint). Early stop at epoch 17, best epoch 13: CER 16.47%, WER 41.07% on validation. The folder is named `extended` because the notebook wrote there.

## 3. Cross dataset (`cross_dataset/`)

No retraining in either direction.

| Train | Test | Samples | CER | WER |
| --- | --- | ---: | ---: | ---: |
| AHL-D | Fidel handwritten | 34,820 | 49.38% | 85.71% |
| Fidel | AHL-D (all splits) | 23,911 | 41.18% | 75.46% |

Both directions fail, so the gap is between the datasets and not one bad training run. Fidel to Fidel at 16.47% shows Fidel can be learned.

## 4. Fine-tuning AHL-D to Fidel (`finetune/`)

Started from the AHL-D checkpoint. Fidel split by writer: 27,855 train / 3,458 val / 3,507 test. 10 epochs, lr 1e-5, best epoch 10 (still improving).

| Set | Samples | CER | WER |
| --- | ---: | ---: | ---: |
| Fidel val | 3,458 | 24.72% | 61.89% |
| Fidel test | 3,507 | 27.97% | 66.98% |
| AHL-D test | 2,990 | 23.69% | 44.57% |

Fidel test CER went from 49.38% to 27.97%. AHL-D CER went from 4.62% (validation) to 23.69% (test), which points to heavy forgetting, but the two numbers are on different splits. Run notebook 06 for the baseline on the test split to get a like for like number.

## Caveats

- The Fidel to Fidel run and the fine-tune used different Fidel splits, so 16.47% and 27.97% are not comparable.
- The zero-shot AHL-D to Fidel number covers all 34,820 Fidel samples, including the writers later used to fine-tune.
- Only one seed (42) per run.

## Not done yet

- AHL-D baseline on test (notebook 06)
- Fine-tuning for more epochs
- Ways to reduce forgetting, for example mixing AHL-D samples into the fine-tune
