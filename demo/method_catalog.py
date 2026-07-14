"""Build the UI method catalog from measured preprocessing benchmarks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILES = (
    ROOT / "outputs" / "testing" / "preprocessing_course_benchmark" / "test_finalists_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_combinations_benchmark" / "test_finalists_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_adaptive_benchmark" / "test_finalists_results.csv",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def _display_name(name: str) -> str:
    replacements = {
        "clahe": "CLAHE",
        "rl": "RL",
        "rgb": "RGB",
        "cv": "CV",
        "nlm": "NLM",
        "3way": "3-Way",
        "2way": "2-Way",
    }
    return " ".join(replacements.get(part, part.capitalize()) for part in name.split("_"))


def _pipeline_steps(config: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    policy = str(config.get("adaptive_policy", "none"))
    if policy != "none":
        steps.append(f"Adaptive routing: {policy.replace('_', ' ')}")
    if bool(config.get("grayscale", False)):
        channel = str(config.get("gray_channel", "luma"))
        steps.append(f"Grayscale ({channel})")
    if bool(config.get("autocontrast", False)):
        steps.append("Global autocontrast")
    if _number(config.get("percentile_high")) > 0:
        steps.append(
            f"Percentile stretch {config.get('percentile_low', 0):g}-{config.get('percentile_high', 100):g}"
        )
    if _number(config.get("clahe_clip_limit")) > 0:
        steps.append(f"CLAHE clip {config.get('clahe_clip_limit'):g}")
    illumination = str(config.get("illumination", "none"))
    if illumination != "none":
        steps.append(illumination.replace("_", " ").title())
    deblur = str(config.get("deblur", "none"))
    if deblur != "none":
        steps.append(deblur.replace("_", " ").title())
    denoise = str(config.get("denoise", "none"))
    if denoise != "none":
        steps.append(denoise.replace("_", " ").title())
    if _number(config.get("sharpen_alpha")) > 0:
        steps.append(f"Mild {config.get('sharpen_method', 'unsharp')} sharpening")
    edge = str(config.get("edge_enhancement", "none"))
    if edge != "none":
        steps.append(f"{edge.replace('_', ' ').title()} edge fusion")
    threshold = str(config.get("threshold", "none"))
    if threshold != "none":
        steps.append(f"{threshold.title()} threshold")
    return steps or ["RGB normalization only"]


def _impact_reason(name: str, config: dict[str, Any]) -> str:
    if str(config.get("adaptive_policy", "none")) != "none":
        return (
            "Routes each crop to a conservative pipeline using measured image quality, "
            "so clean plates are preserved while degraded plates receive stronger correction."
        )
    if "rl_deblur" in name:
        return (
            "Richardson-Lucy restoration recovers softened character strokes, while bilateral "
            "filtering limits amplified noise and protects character boundaries."
        )
    if "clahe" in name:
        return (
            "Local contrast enhancement separates character strokes from uneven plate backgrounds "
            "without applying a strong global exposure shift."
        )
    if "percentile" in name:
        return (
            "Robust percentile stretching ignores extreme pixels and expands the useful tonal range, "
            "making low-contrast strokes easier for the recognizer to attend to."
        )
    if "homomorphic" in name:
        return (
            "Homomorphic filtering suppresses slow illumination changes while retaining high-frequency "
            "character structure under glare, shadows, and uneven lighting."
        )
    if name == "autocontrast":
        return "Expands the global intensity range and improves separation between foreground strokes and background."
    if name == "channel_green":
        return "The green channel often carries cleaner luminance detail and less color noise in camera imagery."
    if name.startswith("gamma"):
        return "A mild nonlinear tone adjustment exposes character detail while avoiding aggressive local enhancement."
    if name == "train_baseline":
        return "Reference pipeline used during fine-tuning; it combines local contrast, edge-preserving denoising, and mild sharpening."
    return "Preserves character geometry while improving the signal presented to PARSeq."


def load_method_catalog(available_configs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return benchmarked runtime configs sorted by measured exact-match accuracy."""
    best_rows: dict[str, dict[str, Any]] = {}
    for csv_path in BENCHMARK_FILES:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        for row in frame.to_dict(orient="records"):
            name = str(row.get("name") or row.get("config") or "").strip()
            if name not in available_configs:
                continue
            candidate = {
                **row,
                "benchmark_source": csv_path.parent.name,
            }
            current = best_rows.get(name)
            if current is None or (
                _number(candidate.get("exact_acc")), _number(candidate.get("char_acc"))
            ) > (_number(current.get("exact_acc")), _number(current.get("char_acc"))):
                best_rows[name] = candidate

    # Keep measured improvements plus the training-time reference pipeline.
    selected = {
        name: row
        for name, row in best_rows.items()
        if _number(row.get("delta_exact")) > 0 or name == "train_baseline"
    }
    if "train_baseline" not in selected and "train_baseline" in available_configs:
        selected["train_baseline"] = {
            "name": "train_baseline",
            "description": available_configs["train_baseline"].description,
            "course_topic": available_configs["train_baseline"].course_topic,
            "exact_acc": 0.0,
            "char_acc": 0.0,
            "delta_exact": 0.0,
            "images_per_second": 0.0,
            "benchmark_source": "runtime_reference",
        }

    ordered = sorted(
        selected.items(),
        key=lambda item: (
            _number(item[1].get("exact_acc")),
            _number(item[1].get("char_acc")),
            _number(item[1].get("images_per_second")),
        ),
        reverse=True,
    )
    catalog: list[dict[str, Any]] = []
    for rank, (name, row) in enumerate(ordered, start=1):
        cfg = available_configs[name]
        cfg_dict = asdict(cfg)
        exact = _number(row.get("exact_acc"))
        char_acc = _number(row.get("char_acc"))
        catalog.append(
            {
                "rank": rank,
                "name": name,
                "display_name": _display_name(name),
                "topic": str(row.get("course_topic") or cfg.course_topic),
                "description": str(row.get("description") or cfg.description),
                "impact_reason": _impact_reason(name, cfg_dict),
                "pipeline_steps": _pipeline_steps(cfg_dict),
                "exact_acc": exact,
                "char_acc": char_acc,
                "delta_exact": _number(row.get("delta_exact")),
                "images_per_second": _number(row.get("images_per_second")),
                "samples": int(_number(row.get("samples"))),
                "is_baseline": name == "train_baseline",
                "benchmark_source": str(row.get("benchmark_source", "")),
            }
        )
    return catalog
