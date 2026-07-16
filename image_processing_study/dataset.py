"""Fixed data split + torch Dataset shared by every method in the comparison.

The single most important control variable in this study is that every
image processing method is trained and evaluated on the *exact same* set of
images per split. ``build_split`` computes that split exactly once (seeded,
deterministic) from ``color_filtered/`` -- callers must not re-shuffle or
re-split per method, only the pixel processing (via a ``methods.Method``)
should vary between runs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from image_processing_study.common import CANVAS_SIZE, MAX_LABEL_LENGTH, normalize_plate_text
from image_processing_study.methods import Method, to_canvas_gray

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["color_filtered/blue", "color_filtered/other", "color_filtered/yellow"]


def list_clean_samples(repo_root: Path = REPO_ROOT) -> list[tuple[Path, str]]:
    """Read (image_path, label) pairs from ``color_filtered/*/labels.txt``.

    Deliberately reimplemented here (instead of importing
    ``rl_deblur.make_dataset.list_clean_samples``) so this module has no
    dependency on ``rl_deblur`` except for the one optional
    ``rl_deblur_restore`` method in ``methods.py``.
    """
    samples: list[tuple[Path, str]] = []
    for rel in SOURCE_DIRS:
        folder = repo_root / rel
        labels_path = folder / "labels.txt"
        if not labels_path.exists():
            continue
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            fname, label = line.split("\t")
            img_path = folder / fname
            if img_path.exists():
                samples.append((img_path, label))
    return samples


def build_split(
    repo_root: Path = REPO_ROOT,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_label_length: int = MAX_LABEL_LENGTH,
) -> dict[str, list[tuple[Path, str]]]:
    """Deterministic train/val/test split, shared by every processing method."""
    samples = list_clean_samples(repo_root)
    cleaned: list[tuple[Path, str]] = []
    for path, label in samples:
        norm = normalize_plate_text(label)
        if 1 <= len(norm) <= max_label_length:
            cleaned.append((path, norm))

    rng = random.Random(seed)
    rng.shuffle(cleaned)

    n = len(cleaned)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    return {
        "val": cleaned[:n_val],
        "test": cleaned[n_val : n_val + n_test],
        "train": cleaned[n_val + n_test :],
    }


class PlateOCRDataset(Dataset):
    """Applies a single :class:`~image_processing_study.methods.Method` on-the-fly.

    Pixel values are normalized to ``[-1, 1]`` (same convention as
    ``rl_deblur``/PARSeq) so a single model architecture accepts input from
    every method unchanged.
    """

    def __init__(self, samples: list[tuple[Path, str]], method: Method):
        self.samples = samples
        self.method = method

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        canvas = to_canvas_gray(image, self.method.resample)
        processed = self.method.process(canvas)
        tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)  # (1, H, W)
        return tensor, label, str(path)


def collate_batch(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(list(images), dim=0), list(labels), list(paths)


assert CANVAS_SIZE == (128, 32), "PlateOCRDataset assumes the (128,32) canvas convention"
