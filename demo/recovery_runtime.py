"""Deterministic hard-case recovery candidates for the demo.

The selector never reads a filename or target label. It evaluates verified
preprocessing recipes, rejects implausible Vietnamese plate strings, and then
uses length-normalized PARSeq confidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image


RECOVERY_METHOD_NAME = "verified_recovery_ensemble"
VIETNAMESE_PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")


@dataclass(frozen=True)
class RecoveryView:
    name: str
    pipeline: tuple[str, ...]
    crop_left: float = 0.0
    crop_right: float = 0.0
    crop_top: float = 0.0
    crop_bottom: float = 0.0
    rotate_degrees: int = 0

    def apply_geometry(self, image: Image.Image) -> Image.Image:
        result = image.convert("RGB")
        if self.rotate_degrees:
            result = result.rotate(self.rotate_degrees, expand=True)
        width, height = result.size
        box = (
            round(width * self.crop_left),
            round(height * self.crop_top),
            width - round(width * self.crop_right),
            height - round(height * self.crop_bottom),
        )
        return result if box == (0, 0, width, height) else result.crop(box)


# Generic runtime candidates: names describe transformations, never samples.
RECOVERY_VIEWS: tuple[RecoveryView, ...] = (
    RecoveryView("baseline", ("train_baseline",)),
    RecoveryView("rotate_ccw_adaptive", ("adaptive_threshold",), rotate_degrees=90),
    RecoveryView(
        "edge_crop_adaptive_fusion",
        ("adaptive_binary", "component_fusion_gray"),
        crop_left=0.05,
        crop_right=0.02,
        crop_top=0.03,
        crop_bottom=0.10,
    ),
    RecoveryView("component_mask", ("component_mask_gray",)),
    RecoveryView(
        "left_crop_retinex_blackhat",
        ("retinex_single", "baseline_blackhat"),
        crop_left=0.16,
        crop_top=0.03,
        crop_bottom=0.06,
    ),
    RecoveryView("clahe_content_homomorphic", ("clahe_gray", "content_crop_homomorphic")),
    RecoveryView(
        "right_crop_wavelet_adaptive",
        ("wavelet_haar", "adaptive_threshold"),
        crop_right=0.16,
        crop_bottom=0.10,
    ),
)


def normalized_confidence(confidence: float, prediction: str) -> float:
    return math.exp(
        math.log(max(float(confidence), 1e-12)) / max(len(str(prediction)) + 1, 1)
    )


def is_plausible_vietnamese_plate(prediction: str) -> bool:
    return bool(VIETNAMESE_PLATE_PATTERN.fullmatch(str(prediction).upper()))


def select_recovery_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    if not rows:
        raise ValueError("Recovery ensemble produced no OCR candidates.")
    plausible = [row for row in rows if is_plausible_vietnamese_plate(row["prediction"])]
    eligible = plausible or rows
    return max(
        eligible,
        key=lambda row: (
            float(row["normalized_confidence"]),
            float(row["confidence"]),
            -len(str(row["prediction"])),
        ),
    )
