# Architecture

CRNN, the same model in every experiment (`ocr/src/models.py`).

- Input: 1x64x512 grayscale line image
- CNN: 6 conv layers (64, 128, 256, 256, 512, 512 channels) with batch norm and ReLU, max pooling after layers 2, 4 and 6
- Height is average pooled to 1, width becomes the sequence axis
- 2 layer bidirectional LSTM, hidden size 256, dropout 0.2
- Linear layer to the characters plus one CTC blank
- CTC loss, greedy decoding
- Vocabulary: 311 characters for AHL-D, 251 for Fidel
