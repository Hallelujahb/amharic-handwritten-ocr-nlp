def greedy_decode(outputs, idx_to_char, blank_idx=0):
    """outputs: (T, B, C) logits from CRNN. Collapses repeats, drops blanks."""
    best = outputs.argmax(dim=2).permute(1, 0)
    texts = []
    for seq in best.tolist():
        chars, prev = [], blank_idx
        for i in seq:
            if i != blank_idx and i != prev:
                chars.append(idx_to_char.get(i, ""))
            prev = i
        texts.append("".join(chars))
    return texts


def decode_targets(targets, lengths, idx_to_char):
    texts, start = [], 0
    for n in lengths.tolist():
        texts.append("".join(idx_to_char.get(v, "") for v in targets[start:start + n].tolist()))
        start += n
    return texts
