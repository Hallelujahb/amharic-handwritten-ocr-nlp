# Evaluation

Numbers are in `results/comparison.csv` and `ocr/research/crnn/README.md`.

Known gaps:

- The AHL-D baseline has not been scored on the AHL-D test split, so the forgetting number is not comparable yet
- The Fidel baseline and the fine-tune used different Fidel splits
- One seed per run
- The fine-tune keeps the AHL-D vocabulary, so Fidel characters outside it cannot be predicted
