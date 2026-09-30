import argparse

import torch

from dataset import load_image
from decode import greedy_decode
from models import CRNN


def load_model(checkpoint_path, device="cpu"):
    ck = torch.load(checkpoint_path, map_location=device)
    idx_to_char = {int(k): v for k, v in ck["idx_to_char"].items()}
    model = CRNN(len(idx_to_char) + 1).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, idx_to_char, ck.get("blank_idx", 0)


@torch.no_grad()
def transcribe(model, idx_to_char, blank_idx, images, device="cpu", batch_size=16):
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([load_image(p) for p in images[i:i + batch_size]]).to(device)
        out += greedy_decode(model(batch), idx_to_char, blank_idx)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transcribe line images with a CRNN checkpoint")
    ap.add_argument("checkpoint")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    m, i2c, blank = load_model(a.checkpoint, a.device)
    for path, text in zip(a.images, transcribe(m, i2c, blank, a.images, a.device)):
        print(f"{path}\t{text}")
