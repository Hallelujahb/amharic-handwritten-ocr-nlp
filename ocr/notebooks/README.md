# Notebooks

Run in Colab with the GPU runtime. Each notebook runs on its own.

| Notebook | What it does |
| --- | --- |
| 01_fidel_crnn_baseline | Fidel handwritten setup, CRNN trained and tested on Fidel |
| 02_ahld_crnn_baseline | AHL-D 29K setup, CRNN trained and tested on AHL-D |
| 03_ahld_to_fidel_eval | AHL-D model on Fidel handwritten, no retraining |
| 04_fidel_to_ahld_eval | Fidel model on AHL-D, no retraining |
| 05_finetune_ahld_to_fidel | Fine-tune the AHL-D model on Fidel, test on Fidel and AHL-D |
| 06_ahld_baseline_test_eval | AHL-D baseline on the AHL-D test split (not run yet) |

`archive/` holds the first AHL-D attempt, replaced by 02.

Paths point to `/content/drive/MyDrive/amharic_htr_research`. Checkpoints are not in the repo, see `checkpoints/` locally.
