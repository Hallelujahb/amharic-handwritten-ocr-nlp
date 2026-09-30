import json
import random

import pandas as pd


def load_fidel_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    df["writer"] = df["writer"].astype(str)
    df["image_filename"] = df["image_filename"].astype(str)
    df["line_text"] = df["line_text"].fillna("").astype(str)
    keep = (df["type"] == "handwritten") & (df["line_text"].str.strip() != "")
    return df[keep].reset_index(drop=True)


def split_by_writer(df, writer_col="writer", fractions=(0.8, 0.1, 0.1), seed=42):
    """Writer-disjoint train/val/test. Same shuffle as the fine-tune notebook."""
    df = df.assign(**{writer_col: df[writer_col].fillna("nan").astype(str)})
    writers = sorted(df[writer_col].unique())
    random.seed(seed)
    random.shuffle(writers)
    n = len(writers)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    groups = [
        set(writers[:n_train]),
        set(writers[n_train:n_train + n_val]),
        set(writers[n_train + n_val:]),
    ]
    parts = [df[df[writer_col].isin(g)].reset_index(drop=True) for g in groups]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
    return parts


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    return {
        "characters": chars,
        "blank_idx": 0,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(i): c for c, i in char_to_idx.items()},
    }


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    v["char_to_idx"] = {c: int(i) for c, i in v["char_to_idx"].items()}
    v["idx_to_char"] = {int(i): c for i, c in v["idx_to_char"].items()}
    return v
