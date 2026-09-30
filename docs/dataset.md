# Datasets

## AHL-D 29K
`misiker/AHLD-29k`. Line images with text. Uses the dataset's own splits: 23,911 train, 3,046 validation, 2,990 test (144, 18 and 18 writers).

## Fidel handwritten
`upanzi/fidel-dataset`, rows with type `handwritten` and non empty text: 34,820 samples.

- Baseline: 31,569 train, 3,251 validation, writer-disjoint (372 and 41 writers)
- Fine-tune: split by writer 80/10/10 with seed 42, giving 27,855 train, 3,458 validation, 3,507 test
- 198 samples have no writer id. They are treated as one writer, so they all land in the same split.

Splits and vocabularies are in `ocr/data`.
