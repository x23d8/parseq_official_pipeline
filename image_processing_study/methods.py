"""Registry of image processing methods compared in this study.

Each :class:`Method` bundles a resize filter (applied once, when the raw crop
is rendered onto the fixed 32x128 canvas) and a ``process`` function (applied
to the resulting uint8 grayscale canvas). Every method in ``CORE_METHODS``
runs on plain numpy/OpenCV/scikit-image so the comparison does not secretly
depend on any learned component -- except ``rl_deblur_restore``, which is
intentionally the one deep-learning entry and is loaded lazily via
:func:`try_build_rl_deblur_method` so a missing checkpoint degrades to
"skip this method" instead of crashing the whole sweep.

All methods are deterministic (no randomness) so repeated runs on the same
image are reproducible -- important since every method is compared against
the exact same fixed train/val/test split (see ``dataset.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from image_processing_study.common import CANVAS_SIZE

logger = logging.getLogger(__name__)

ProcessFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Method:
    name: str
    chapter: str
    description: str
    process: ProcessFn
    resample: int = Image.BILINEAR


def _identity(arr: np.ndarray) -> np.ndarray:
    return arr


def to_canvas_gray(image: Image.Image, resample: int = Image.BILINEAR) -> np.ndarray:
    resized = image.convert("L").resize(CANVAS_SIZE, resample)
    return np.asarray(resized, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1.1 / 7.1 -- gray-level processing & sampling/interpolation
# ---------------------------------------------------------------------------


def _hist_eq(arr: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(arr)


# ---------------------------------------------------------------------------
# 1.2 -- basic binary image processing
# ---------------------------------------------------------------------------


def _otsu_binary(arr: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# 2.1 -- linear filtering / contrast enhancement
# ---------------------------------------------------------------------------


def _clahe(arr: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


# ---------------------------------------------------------------------------
# 2.2 -- nonlinear filtering
# ---------------------------------------------------------------------------


def _median_denoise(arr: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(arr, 3)


def _bilateral_denoise(arr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=5)


# ---------------------------------------------------------------------------
# 2.3 -- morphological filtering
# ---------------------------------------------------------------------------


def _morph_tophat(arr: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_tophat = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, kernel)
    black_tophat = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, kernel)
    enhanced = arr.astype(np.int16) + white_tophat.astype(np.int16) - black_tophat.astype(np.int16)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.5 -- filtering in the frequency domain
# ---------------------------------------------------------------------------


def _fft_gaussian_lowpass_mask(shape: tuple[int, int], cutoff_ratio: float = 0.15) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows / 2.0, cols / 2.0
    y, x = np.ogrid[:rows, :cols]
    dist2 = (x - ccol) ** 2 + (y - crow) ** 2
    d0 = cutoff_ratio * min(rows, cols)
    return np.exp(-dist2 / (2.0 * (d0**2)))


def _freq_highboost(arr: np.ndarray, boost: float = 1.5) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(arr.astype(np.float64)))
    lowpass = _fft_gaussian_lowpass_mask(arr.shape, cutoff_ratio=0.15)
    highboost_mask = 1.0 + boost * (1.0 - lowpass)
    filtered = np.fft.ifft2(np.fft.ifftshift(f * highboost_mask))
    return np.clip(np.abs(filtered), 0, 255).astype(np.uint8)


def _homomorphic(arr: np.ndarray, gamma_low: float = 0.5, gamma_high: float = 1.5) -> np.ndarray:
    """Homomorphic filtering: log domain -> FFT high-emphasis -> exp domain.

    Suppresses the low-frequency illumination component while boosting the
    high-frequency reflectance (edge/text) component, which helps unevenly
    lit plate crops without needing a learned model.
    """
    img_log = np.log1p(arr.astype(np.float64))
    f = np.fft.fftshift(np.fft.fft2(img_log))
    lowpass = _fft_gaussian_lowpass_mask(arr.shape, cutoff_ratio=0.2)
    h = (gamma_high - gamma_low) * (1.0 - lowpass) + gamma_low
    filtered = np.fft.ifft2(np.fft.ifftshift(f * h))
    img_exp = np.expm1(np.real(filtered))
    img_exp -= img_exp.min()
    denom = img_exp.max()
    if denom > 1e-6:
        img_exp = img_exp / denom * 255.0
    return np.clip(img_exp, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3.5 / Ch.4 -- wavelet denoising
# ---------------------------------------------------------------------------


def _wavelet_denoise(arr: np.ndarray) -> np.ndarray:
    from skimage.restoration import denoise_wavelet

    img_float = arr.astype(np.float64) / 255.0
    denoised = denoise_wavelet(img_float, method="BayesShrink", mode="soft", rescale_sigma=True)
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3.1 / 3.4 / 3.6 -- restoration process model & MMSE (Wiener) filtering
# ---------------------------------------------------------------------------


def gaussian_psf(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float64)


def wiener_deconvolve(arr: np.ndarray, psf: np.ndarray, balance: float = 0.1) -> np.ndarray:
    from skimage.restoration import wiener as skimage_wiener

    img_float = arr.astype(np.float64) / 255.0
    restored = skimage_wiener(img_float, psf, balance=balance, clip=True)
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def _wiener_restore_assumed_psf(arr: np.ndarray) -> np.ndarray:
    """Experiment A variant: the real plate crops have no known degradation
    kernel, so this assumes a mild Gaussian PSF and uses Wiener deconvolution
    as a general-purpose restoration/sharpening operator. Experiment B
    (``degrade.py`` / ``run_experiment_b.py``) instead deconvolves with the
    *true* PSF used to synthesize the degradation, which is the textbook use
    of Wiener/MMSE restoration (Ch.3.1/3.6): a known degradation model.
    """
    return wiener_deconvolve(arr, gaussian_psf(size=5, sigma=1.0), balance=0.1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CORE_METHODS: list[Method] = [
    Method("raw", "baseline", "No processing (control group), bilinear resize.", _identity, Image.BILINEAR),
    Method("bicubic_resize", "7.1 Sampling & Interpolation", "Bicubic instead of bilinear resize to the canvas.", _identity, Image.BICUBIC),
    Method("hist_eq", "1.1 Gray-level processing", "Global histogram equalization.", _hist_eq),
    Method("otsu_binary", "1.2 Binary image processing", "Otsu global thresholding to a binary image.", _otsu_binary),
    Method("clahe", "2.1 Linear filtering / enhancement", "Contrast-limited adaptive histogram equalization.", _clahe),
    Method("median_denoise", "2.2 Nonlinear filtering", "Median filter (impulse-noise robust smoothing).", _median_denoise),
    Method("bilateral_denoise", "2.2 Nonlinear filtering", "Bilateral filter (edge-preserving smoothing).", _bilateral_denoise),
    Method("morph_tophat", "2.3 Morphological filtering", "White top-hat + black top-hat contrast boost.", _morph_tophat),
    Method("freq_highboost", "2.5 Frequency-domain filtering", "High-boost filtering via FFT Gaussian high-pass mask.", _freq_highboost),
    Method("homomorphic", "2.5 / 3.1 Restoration model", "Homomorphic filtering (illumination/reflectance separation).", _homomorphic),
    Method("wavelet_denoise", "3.5 Wavelet denoising", "Wavelet-domain soft-threshold denoising (BayesShrink).", _wavelet_denoise),
    Method("wiener_restore", "3.1/3.4/3.6 Restoration & MMSE filtering", "Wiener deconvolution with an assumed mild Gaussian PSF.", _wiener_restore_assumed_psf),
]

CORE_METHOD_NAMES = [m.name for m in CORE_METHODS]


def get_core_method(name: str) -> Method:
    for method in CORE_METHODS:
        if method.name == name:
            return method
    raise KeyError(f"Unknown image processing method: {name}")


def try_build_rl_deblur_method(checkpoint: str | Path, device: str = "cpu") -> Method | None:
    """Wrap the pretrained ``rl_deblur`` PixelRL agent as a comparison Method.

    Returns ``None`` (with a warning log, not an exception) if the checkpoint
    is missing or fails to load, so a sweep over ``CORE_METHODS +
    [rl_deblur_restore]`` can gracefully drop this one entry instead of
    crashing the whole comparison.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        logger.warning("rl_deblur checkpoint not found at %s -- skipping rl_deblur_restore method.", checkpoint)
        return None
    try:
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch

        from rl_deblur import env as rl_env
        from rl_deblur.model import FCNActorCritic

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        agent = FCNActorCritic(channels=cfg["channels"], rmc_kernel_size=cfg.get("rmc_kernel_size", 9))
        agent.load_state_dict(ckpt["model_state_dict"])
        agent.to(device)
        agent.eval()
        num_steps = int(cfg["num_steps"])
    except Exception:
        logger.exception("Failed to load rl_deblur checkpoint from %s -- skipping rl_deblur_restore method.", checkpoint)
        return None

    @torch.no_grad()
    def _rl_restore(arr: np.ndarray) -> np.ndarray:
        state = arr.astype(np.float32)[None, ...]
        for _ in range(num_steps):
            state_t = torch.from_numpy(state / 255.0).unsqueeze(1).float().to(device)
            logits, _ = agent(state_t)
            action_map = logits.argmax(dim=1).cpu().numpy()
            state, _ = rl_env.step(state, action_map, state)  # clean unused, action-apply only
        return np.clip(state[0], 0, 255).astype(np.uint8)

    return Method(
        "rl_deblur_restore",
        "Deep-learning restoration (comparison)",
        "PixelRL + A2C agent from rl_deblur/, applied greedily.",
        _rl_restore,
    )


def build_registry(include_rl: bool, rl_checkpoint: str | Path = "", device: str = "cpu") -> list[Method]:
    methods = list(CORE_METHODS)
    if include_rl:
        rl_method = try_build_rl_deblur_method(rl_checkpoint, device=device)
        if rl_method is not None:
            methods.append(rl_method)
    return methods
