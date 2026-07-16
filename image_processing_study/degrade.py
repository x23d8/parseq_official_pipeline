"""Synthetic, ground-truth degradation for Experiment B (restoration study).

Real plate photos have no known "clean" counterpart, so Experiment A cannot
measure classical restoration quality (Ch.3 needs a known degradation model
to be meaningful). This module degrades the *same fixed test split* used by
Experiment A (``dataset.build_split`` with the shared ``ocr_train.SPLIT_SEED``)
with a known blur kernel or known noise level, so PSNR/SSIM against the
original clean pixels -- and a textbook Wiener deconvolution using the
*true* PSF -- are both possible.

Everything is computed in-memory and on-the-fly (no dataset files written to
disk), mirroring the on-the-fly philosophy of ``dataset.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_processing_study import ocr_train
from image_processing_study.dataset import build_split
from image_processing_study.methods import to_canvas_gray


@dataclass(frozen=True)
class DegradationSpec:
    kind: str  # "gaussian_blur" | "defocus_blur" | "motion_blur" | "gaussian_noise"
    strength: float
    angle: float = 0.0  # only used by motion_blur


def gaussian_kernel_2d(sigma: float, ksize: int | None = None) -> np.ndarray:
    if ksize is None:
        ksize = int(sigma * 3) | 1
    ksize = max(ksize, 3)
    k1d = cv2.getGaussianKernel(ksize, sigma)
    return (k1d @ k1d.T).astype(np.float64)


def motion_kernel_2d(length: int, angle: float) -> np.ndarray:
    length = max(length, 3)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    kernel_sum = kernel.sum()
    if kernel_sum < 1e-6:
        kernel[length // 2, length // 2] = 1.0
        kernel_sum = 1.0
    return (kernel / kernel_sum).astype(np.float64)


def random_degradation_spec(rng: random.Random) -> DegradationSpec:
    """Mild, recoverable degradation levels (same rationale as
    ``rl_deblur/make_dataset.py``: strong enough to hurt OCR, mild enough
    that character strokes are not fully destroyed).
    """
    kind = rng.choice(["gaussian_blur", "defocus_blur", "motion_blur", "gaussian_noise"])
    if kind == "gaussian_blur":
        return DegradationSpec(kind, rng.uniform(0.8, 1.8))
    if kind == "defocus_blur":
        return DegradationSpec(kind, rng.uniform(1.5, 3.0))
    if kind == "motion_blur":
        return DegradationSpec(kind, rng.uniform(4, 9), angle=rng.uniform(0, 180))
    return DegradationSpec("gaussian_noise", rng.uniform(10, 25))  # noise sigma in pixel units


def apply_degradation(clean: np.ndarray, spec: DegradationSpec) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (degraded_u8, psf) -- psf is the *exact* kernel used for blur
    kinds (so Experiment B can deconvolve with the true degradation
    function), or ``None`` for ``gaussian_noise`` (no blur kernel involved).
    """
    clean_f = clean.astype(np.float64)
    if spec.kind in ("gaussian_blur", "defocus_blur"):
        psf = gaussian_kernel_2d(spec.strength)
        degraded = cv2.filter2D(clean_f, -1, psf, borderType=cv2.BORDER_REFLECT)
        return np.clip(degraded, 0, 255).astype(np.uint8), psf
    if spec.kind == "motion_blur":
        psf = motion_kernel_2d(int(spec.strength), spec.angle)
        degraded = cv2.filter2D(clean_f, -1, psf, borderType=cv2.BORDER_REFLECT)
        return np.clip(degraded, 0, 255).astype(np.uint8), psf
    if spec.kind == "gaussian_noise":
        noisy = clean_f + np.random.normal(0.0, spec.strength, size=clean_f.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8), None
    raise ValueError(spec.kind)


def build_degraded_records(
    seed: int = 123,
    split_seed: int = ocr_train.SPLIT_SEED,
    limit: int | None = None,
) -> list[dict]:
    """One record per test-split image: clean canvas, degraded canvas, and
    the degradation spec/PSF used to produce it.
    """
    split = build_split(seed=split_seed)
    test_samples = split["test"]
    if limit is not None:
        test_samples = test_samples[:limit]

    rng = random.Random(seed)
    np.random.seed(seed)  # apply_degradation's gaussian_noise branch uses np.random

    records = []
    for path, label in test_samples:
        image = Image.open(path).convert("RGB")
        clean = to_canvas_gray(image, resample=Image.BILINEAR)
        spec = random_degradation_spec(rng)
        degraded, psf = apply_degradation(clean, spec)
        records.append(
            {
                "image_path": str(path),
                "label": label,
                "clean": clean,
                "degraded": degraded,
                "kind": spec.kind,
                "strength": spec.strength,
                "psf": psf,
            }
        )
    return records
