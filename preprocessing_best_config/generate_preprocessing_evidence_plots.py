"""Generate evidence-focused figures for the preprocessing presentation.

The figures deliberately use test samples for which the saved train baseline
prediction is wrong and at least one evaluated preprocessing method is right.
Besides enlarged outputs, the script plots amplified difference/high-frequency
maps so that mild CLAHE and bilateral filtering are visible and measurable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocessing import (
    _richardson_lucy,
    get_preprocessing_config,
    preprocess_plate_image,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "preprocessing_best_config" / "report_images"
ADAPTIVE_DIR = ROOT / "outputs" / "testing" / "preprocessing_adaptive_benchmark"
COURSE_DIR = ROOT / "outputs" / "testing" / "preprocessing_course_benchmark"
COMBO_DIR = ROOT / "outputs" / "testing" / "preprocessing_combinations_benchmark"


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {Path(row["image_path"]).name: row for row in csv.DictReader(handle)}


def read_routes(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {Path(row["image_path"]).name: row for row in csv.DictReader(handle)}


def load_image(path_text: str) -> Image.Image:
    path = Path(path_text)
    if not path.exists():
        # Saved CSVs contain absolute Windows paths; this fallback keeps the
        # script reproducible if the repository is moved as a whole.
        marker = "dataset\\"
        relative = path_text.lower().split(marker, 1)[-1]
        path = ROOT / "dataset" / Path(relative.replace("\\", "/"))
    return Image.open(path).convert("RGB")


def gray_array(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def show_plate(ax: plt.Axes, image: Image.Image | np.ndarray, title: str) -> None:
    array = np.asarray(image)
    if array.ndim == 2:
        ax.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(array, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")


def local_std(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    work = image.astype(np.float32)
    mean = cv2.boxFilter(work, -1, (kernel_size, kernel_size), normalize=True)
    mean_sq = cv2.boxFilter(work * work, -1, (kernel_size, kernel_size), normalize=True)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def hf_residual(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    return np.abs(image.astype(np.float32) - blur.astype(np.float32))


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    work = image.astype(np.float32)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def prediction_title(name: str, row: dict[str, str], correct: bool) -> str:
    symbol = "CORRECT" if correct else "WRONG"
    confidence = float(row["confidence"])
    return f"{name}\n{row['prediction']}  [{symbol}]\nconf={confidence:.2f}"


def corrected_cases_figure(
    baseline: dict[str, dict[str, str]],
    clahe: dict[str, dict[str, str]],
    combo: dict[str, dict[str, str]],
    adaptive: dict[str, dict[str, str]],
    routes: dict[str, dict[str, str]],
) -> None:
    names = [
        "Dieu_0480_p02_02.png",
        "Hung_0218_p02_02.png",
        "Dieu_0049_p02_02.png",
        "general_000653_quandoi.jpg",
        "Dieu_0194_p02_02.png",
    ]
    configs = [
        ("Train baseline", "train_baseline", baseline),
        ("Gentle CLAHE", "clahe_clip1_tile4", clahe),
        ("CLAHE + RL + BF", "clahe_rl_deblur_bilateral", combo),
        ("Adaptive router", "adaptive_noise_3way", adaptive),
    ]
    fig, axes = plt.subplots(len(names), 5, figsize=(17, 13), constrained_layout=True)
    for row_idx, filename in enumerate(names):
        source_row = baseline[filename]
        target = source_row["target"]
        image = load_image(source_row["image_path"])
        route = routes[filename]
        show_plate(
            axes[row_idx, 0],
            image,
            f"Original: {filename}\nGT={target}; noise={float(route['noise']):.1f}",
        )
        for col_idx, (label, config_name, rows) in enumerate(configs, start=1):
            output = preprocess_plate_image(image, config_name)
            pred_row = rows[filename]
            is_correct = pred_row["prediction"] == target
            title = prediction_title(label, pred_row, is_correct)
            if config_name == "adaptive_noise_3way":
                title += f"\n{route['selected_pipeline']}"
            show_plate(axes[row_idx, col_idx], output, title)
            axes[row_idx, col_idx].title.set_color("#147d2a" if is_correct else "#b51f1f")

    fig.suptitle(
        "Saved test evidence: baseline errors corrected by evaluated preprocessing",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "presentation_corrected_hard_cases.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def clahe_evidence_figure(baseline: dict[str, dict[str, str]]) -> None:
    filename = "Dieu_0480_p02_02.png"
    row = baseline[filename]
    image = load_image(row["image_path"])
    original = gray_array(image)
    config = get_preprocessing_config("clahe_clip1_tile4")
    clahe = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=(config.clahe_tile_size, config.clahe_tile_size),
    ).apply(original)
    difference = np.abs(clahe.astype(np.int16) - original.astype(np.int16)).astype(np.float32)
    std_original = local_std(original)
    std_clahe = local_std(clahe)
    p_range_original = float(np.percentile(original, 95) - np.percentile(original, 5))
    p_range_clahe = float(np.percentile(clahe, 95) - np.percentile(clahe, 5))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    show_plate(axes[0, 0], original, f"Original grayscale\nP95-P5={p_range_original:.1f}")
    show_plate(axes[0, 1], clahe, f"CLAHE 1.0 / 4x4\nP95-P5={p_range_clahe:.1f}")
    im = axes[0, 2].imshow(np.clip(difference * 4.0, 0, 255), cmap="magma", vmin=0, vmax=255)
    axes[0, 2].set_title(f"|CLAHE - gray| x4\nmean abs change={difference.mean():.1f}", fontsize=10)
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.03)

    axes[1, 0].hist(original.ravel(), bins=64, range=(0, 255), alpha=0.62, label="gray")
    axes[1, 0].hist(clahe.ravel(), bins=64, range=(0, 255), alpha=0.55, label="CLAHE")
    axes[1, 0].set_title("Intensity distributions")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Pixel count")
    axes[1, 0].legend()
    contrast_max = float(max(np.percentile(std_original, 99), np.percentile(std_clahe, 99)))
    axes[1, 1].imshow(std_original, cmap="viridis", vmin=0, vmax=contrast_max)
    axes[1, 1].set_title(f"Local contrast before\nmean local std={std_original.mean():.1f}")
    axes[1, 1].axis("off")
    im2 = axes[1, 2].imshow(std_clahe, cmap="viridis", vmin=0, vmax=contrast_max)
    axes[1, 2].set_title(f"Local contrast after\nmean local std={std_clahe.mean():.1f}")
    axes[1, 2].axis("off")
    fig.colorbar(im2, ax=axes[1, 1:], fraction=0.025, pad=0.02, label="Local std")
    fig.suptitle(
        "CLAHE evidence on a corrected low-contrast case (GT 77X79442)",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "presentation_clahe_measurable_evidence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def denoising_evidence_figure(baseline: dict[str, dict[str, str]]) -> None:
    filename = "Hung_0218_p02_02.png"
    row = baseline[filename]
    image = load_image(row["image_path"])
    original = gray_array(image)
    config = get_preprocessing_config("rl_deblur_bilateral_lowpass")
    after_rl = _richardson_lucy(
        original,
        config.deblur_kernel_size,
        config.deblur_sigma,
        config.deblur_iterations,
    )
    final = cv2.bilateralFilter(
        after_rl,
        config.bilateral_d,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )
    removed = np.abs(after_rl.astype(np.int16) - final.astype(np.int16)).astype(np.float32)
    hf_before = hf_residual(after_rl)
    hf_after = hf_residual(final)
    gradient_before = sobel_magnitude(after_rl)
    gradient_after = sobel_magnitude(final)
    smooth_mask = gradient_before <= np.percentile(gradient_before, 50)
    edge_mask = gradient_before >= np.percentile(gradient_before, 85)
    smooth_hf_before = float(hf_before[smooth_mask].mean())
    smooth_hf_after = float(hf_after[smooth_mask].mean())
    noise_reduction = 100.0 * (1.0 - smooth_hf_after / max(smooth_hf_before, 1e-6))
    edge_retention = 100.0 * float(gradient_after[edge_mask].mean()) / max(
        float(gradient_before[edge_mask].mean()), 1e-6
    )
    hf_max = float(max(np.percentile(hf_before, 99), np.percentile(hf_after, 99)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    show_plate(axes[0, 0], original, "Observed grayscale\nnoise_score=12 (high branch)")
    show_plate(axes[0, 1], after_rl, "After Richardson-Lucy x3\nrestores edges, may amplify ringing")
    show_plate(axes[0, 2], final, "After bilateral d=3, 25/25\nrouter prediction: 51F92687 [CORRECT]")
    im = axes[1, 0].imshow(np.clip(removed * 10.0, 0, 255), cmap="magma", vmin=0, vmax=255)
    axes[1, 0].set_title(f"|RL - bilateral| x10\nmean removed magnitude={removed.mean():.2f}")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.03)
    axes[1, 1].imshow(hf_before, cmap="inferno", vmin=0, vmax=hf_max)
    axes[1, 1].set_title(f"High-frequency residual before\nsmooth-region mean={smooth_hf_before:.2f}")
    axes[1, 1].axis("off")
    im2 = axes[1, 2].imshow(hf_after, cmap="inferno", vmin=0, vmax=hf_max)
    axes[1, 2].set_title(f"High-frequency residual after\nsmooth-region mean={smooth_hf_after:.2f}")
    axes[1, 2].axis("off")
    fig.colorbar(im2, ax=axes[1, 1:], fraction=0.025, pad=0.02, label="|I - GaussianBlur3x3(I)|")
    fig.suptitle(
        f"Bilateral denoising evidence: {noise_reduction:.1f}% less smooth-region HF residual, "
        f"{edge_retention:.1f}% strong-edge retention",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "presentation_bilateral_denoising_evidence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = read_predictions(ADAPTIVE_DIR / "predictions_test_train_baseline.csv")
    adaptive = read_predictions(ADAPTIVE_DIR / "predictions_test_adaptive_noise_3way.csv")
    clahe = read_predictions(COURSE_DIR / "predictions_test_clahe_clip1_tile4.csv")
    combo = read_predictions(COMBO_DIR / "predictions_test_clahe_rl_deblur_bilateral.csv")
    routes = read_routes(ADAPTIVE_DIR / "adaptive_noise_routing_test.csv")
    corrected_cases_figure(baseline, clahe, combo, adaptive, routes)
    clahe_evidence_figure(baseline)
    denoising_evidence_figure(baseline)
    print("Generated evidence figures in", OUT_DIR)


if __name__ == "__main__":
    main()
