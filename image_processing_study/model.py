"""Lightweight CRNN + CTC OCR model used as the shared "base model" for the study.

Deliberately a small, from-scratch architecture (not PARSeq): the whole point
of this comparison is to reveal accuracy differences *caused by* the input
image processing method, and a strong pretrained transformer like PARSeq is
good enough to mostly shrug those differences off. A small CRNN (Shi et al.,
"An End-to-End Trainable Neural Network for Image-based Sequence
Recognition") trained from scratch on ~2900 images is far more sensitive to
input quality.

The CNN backbone follows the original CRNN pooling schedule (channels halved
here) so a fixed 32x128 input always collapses to exactly 1x31 before the
recurrent layers -- every sample in a batch therefore has the *same* CTC
input length, so no length padding/masking is needed anywhere in this
module. That sidesteps one of the most common sources of CTC training bugs.
"""

from __future__ import annotations

import torch
from torch import nn

from image_processing_study.common import ANPR_CHARSET

BLANK_IDX = 0
NUM_CLASSES = len(ANPR_CHARSET) + 1  # +1 for the CTC blank


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32
            nn.Conv2d(256, 256, 2, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 2x32 -> 1x31
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.head = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, 32, 128) -> log_probs: (T=31, B, num_classes)."""
        features = self.cnn(images)  # (B, C, 1, T)
        b, c, h, w = features.shape
        assert h == 1, f"expected CNN backbone to collapse height to 1, got {h}"
        features = features.squeeze(2)  # (B, C, T)
        features = features.permute(2, 0, 1)  # (T, B, C)
        recurrent, _ = self.rnn(features)  # (T, B, 2*hidden)
        logits = self.head(recurrent)  # (T, B, num_classes)
        return logits.log_softmax(dim=2)

    @property
    def time_steps(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 128)
            return self.cnn(dummy).shape[-1]


def ctc_greedy_decode(log_probs: torch.Tensor) -> tuple[list[str], list[float]]:
    """log_probs: (T, B, C) -> (predicted strings, mean-confidence per string)."""
    max_log_probs, max_indices = log_probs.max(dim=2)  # (T, B)
    max_indices = max_indices.transpose(0, 1).cpu().tolist()  # (B, T)
    max_log_probs = max_log_probs.transpose(0, 1).cpu().tolist()  # (B, T)

    texts: list[str] = []
    confidences: list[float] = []
    for indices, log_ps in zip(max_indices, max_log_probs):
        chars = []
        kept_log_ps = []
        prev = -1
        for idx, lp in zip(indices, log_ps):
            if idx != prev and idx != BLANK_IDX:
                chars.append(ANPR_CHARSET[idx - 1])
                kept_log_ps.append(lp)
            prev = idx
        texts.append("".join(chars))
        confidences.append(float(torch.tensor(kept_log_ps).exp().mean()) if kept_log_ps else 0.0)
    return texts, confidences


def encode_targets(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """CTC target encoding: concatenated target indices + per-sample lengths."""
    char_to_idx = {ch: i + 1 for i, ch in enumerate(ANPR_CHARSET)}
    flat: list[int] = []
    lengths: list[int] = []
    for label in labels:
        idxs = [char_to_idx[ch] for ch in label if ch in char_to_idx]
        flat.extend(idxs)
        lengths.append(len(idxs))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
