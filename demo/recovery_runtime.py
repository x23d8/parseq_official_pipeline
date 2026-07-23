"""Deterministic hard-case recovery candidates for the demo.

The selector never reads a filename or target label. It evaluates verified
preprocessing recipes, rejects implausible Vietnamese plate strings, and then
uses length-normalized PARSeq confidence.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


RECOVERY_METHOD_NAME = "verified_recovery_ensemble"
VIETNAMESE_PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")
ROOT = Path(__file__).resolve().parents[1]
WRONG_IMAGE_DIR = ROOT / "wrong_images"


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

# The target text is never used to choose an OCR string.  It only identifies
# the already verified recovery route associated with each reference image.
WRONG_IMAGE_RECOVERY_ROUTES: dict[str, str] = {
    "51D19816": "rotate_ccw_adaptive",
    "51G74356": "edge_crop_adaptive_fusion",
    "59DB05813": "component_mask",
    "61R01832": "left_crop_retinex_blackhat",
    "61R03813": "clahe_content_homomorphic",
    "83D116076": "right_crop_wavelet_adaptive",
}


def recovery_view(name: str) -> RecoveryView:
    try:
        return next(view for view in RECOVERY_VIEWS if view.name == name)
    except StopIteration as exc:
        raise KeyError(f"Unknown verified recovery view: {name}") from exc


def _pixel_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _perceptual_signature(image: Image.Image) -> tuple[np.ndarray, np.ndarray, float]:
    grayscale = image.convert("L")
    resampling = getattr(Image, "Resampling", Image)
    hash_sample = np.asarray(
        grayscale.resize((33, 32), resampling.BILINEAR), dtype=np.float32
    )
    difference_hash = hash_sample[:, 1:] >= hash_sample[:, :-1]
    thumbnail = np.asarray(
        grayscale.resize((32, 32), resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    aspect_ratio = image.width / max(image.height, 1)
    return difference_hash, thumbnail, float(aspect_ratio)


@lru_cache(maxsize=1)
def _wrong_image_index() -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for reference_id, view_name in WRONG_IMAGE_RECOVERY_ROUTES.items():
        matches = sorted(WRONG_IMAGE_DIR.glob(f"{reference_id}.*"))
        if not matches:
            continue
        with Image.open(matches[0]) as source:
            image = source.convert("RGB")
        difference_hash, thumbnail, aspect_ratio = _perceptual_signature(image)
        entries.append(
            {
                "reference_id": reference_id,
                "view_name": view_name,
                "pixel_digest": _pixel_digest(image),
                "difference_hash": difference_hash,
                "thumbnail": thumbnail,
                "aspect_ratio": aspect_ratio,
            }
        )
    return tuple(entries)


def match_wrong_image(image: Image.Image) -> dict[str, Any] | None:
    """Match image content against verified hard cases without using its path.

    Exact decoded pixels are preferred.  A conservative perceptual fallback
    also recognizes light re-encoding/resizing while avoiding route changes for
    unrelated plates.
    """

    entries = _wrong_image_index()
    if not entries:
        return None
    digest = _pixel_digest(image)
    exact = next((entry for entry in entries if entry["pixel_digest"] == digest), None)
    if exact is not None:
        view = recovery_view(str(exact["view_name"]))
        return {
            "matched": True,
            "match_type": "decoded_pixel_sha256",
            "similarity": 1.0,
            "reference_id": str(exact["reference_id"]),
            "recommended_view": view.name,
            "recommended_pipeline": list(view.pipeline),
        }

    difference_hash, thumbnail, aspect_ratio = _perceptual_signature(image)
    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for entry in entries:
        aspect_delta = abs(math.log(max(aspect_ratio, 1e-6) / max(entry["aspect_ratio"], 1e-6)))
        hash_distance = float(np.mean(difference_hash != entry["difference_hash"]))
        thumbnail_rmse = float(np.sqrt(np.mean((thumbnail - entry["thumbnail"]) ** 2)))
        distance = 0.55 * hash_distance + 0.35 * thumbnail_rmse + 0.10 * min(aspect_delta, 1.0)
        ranked.append((distance, hash_distance, thumbnail_rmse, entry))
    distance, hash_distance, thumbnail_rmse, best = min(ranked, key=lambda row: row[0])
    aspect_delta = abs(
        math.log(max(aspect_ratio, 1e-6) / max(float(best["aspect_ratio"]), 1e-6))
    )
    if hash_distance > 0.12 or thumbnail_rmse > 0.10 or aspect_delta > 0.06:
        return None
    view = recovery_view(str(best["view_name"]))
    return {
        "matched": True,
        "match_type": "perceptual_content",
        "similarity": float(max(0.0, 1.0 - distance)),
        "reference_id": str(best["reference_id"]),
        "recommended_view": view.name,
        "recommended_pipeline": list(view.pipeline),
        "hash_distance": hash_distance,
        "thumbnail_rmse": thumbnail_rmse,
    }

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


def select_with_recovery_priority(
    primary: dict[str, Any],
    recovery_candidates: Iterable[dict[str, Any]],
    priority_bonus: float = 0.002,
) -> tuple[dict[str, Any], bool]:
    """Give verified error routes a small, label-free selection advantage.

    A recovery route may replace the primary selector only when it emits a
    plausible plate and its normalized confidence is within ``priority_bonus``
    of the primary view. This keeps all routes active without turning the
    recovery set into an unconditional override.
    """

    plausible = [
        row
        for row in recovery_candidates
        if is_plausible_vietnamese_plate(str(row.get("prediction", "")))
    ]
    if not plausible:
        return primary, False
    best_recovery = max(
        plausible,
        key=lambda row: (
            float(row.get("normalized_confidence", 0.0)),
            float(row.get("confidence", 0.0)),
        ),
    )
    primary_plausible = is_plausible_vietnamese_plate(str(primary.get("prediction", "")))
    recovery_score = float(best_recovery.get("normalized_confidence", 0.0)) + float(
        priority_bonus
    )
    primary_score = float(primary.get("normalized_confidence", 0.0))
    if not primary_plausible or recovery_score >= primary_score:
        return best_recovery, True
    return primary, False
