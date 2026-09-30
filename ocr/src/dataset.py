import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_H, IMG_W = 64, 512


def load_image(path_or_img):
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = img.convert("L").resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


class LineDataset(Dataset):
    """samples: list of (image path or PIL image, text). Text is encoded with char_to_idx."""

    def __init__(self, samples, char_to_idx):
        self.samples = samples
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, text = self.samples[i]
        target = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return load_image(img), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return images, targets, lengths
