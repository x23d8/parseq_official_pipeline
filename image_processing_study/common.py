"""Shared constants and small pure-Python helpers used across this module.

Kept separate (and independent from ``parseq/``) so ``image_processing_study``
does not need to import the heavy vendored PARSeq package just to reuse a
charset or an edit-distance function.
"""

from __future__ import annotations

ANPR_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANVAS_SIZE = (128, 32)  # (width, height), matches rl_deblur/PARSeq img_size=(32,128) convention
MAX_LABEL_LENGTH = 12


def normalize_plate_text(text: object) -> str:
    return "".join(ch for ch in str(text).upper() if ch in ANPR_CHARSET)


def edit_distance(left: str, right: str) -> int:
    left = normalize_plate_text(left)
    right = normalize_plate_text(right)
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        for j, rc in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (lc != rc)))
        previous = current
    return previous[-1]
