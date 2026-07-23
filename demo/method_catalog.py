"""Build a compact UI catalog from measured preprocessing benchmarks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .recovery_runtime import RECOVERY_METHOD_NAME
except ImportError:
    from recovery_runtime import RECOVERY_METHOD_NAME


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILES = (
    ROOT / "outputs" / "testing" / "preprocessing_course_benchmark" / "validation_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_combinations_benchmark" / "validation_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_adaptive_benchmark" / "validation_results.csv",
)
TEST_BENCHMARK_FILES = (
    ROOT / "outputs" / "testing" / "preprocessing_course_benchmark" / "test_finalists_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_combinations_benchmark" / "test_finalists_results.csv",
    ROOT / "outputs" / "testing" / "preprocessing_adaptive_benchmark" / "test_finalists_results.csv",
)

# Expose reusable blocks instead of all pre-baked benchmark combinations.
# PixelRL and complete learned selectors are appended below.
UI_METHOD_NAMES = (
    "clahe_gray",
    "wavelet_haar",
    "bilateral_denoise",
    "median_denoise",
    "gaussian_denoise",
    "nlm_denoise",
    "homomorphic_filter",
    "retinex_multiscale",
    "wiener_deconv",
    "richardson_lucy_deblur",
    "freq_highboost",
    "unsharp_mild",
    "morph_tophat",
    "otsu_binary",
    "adaptive_binary",
    "autocontrast",
    "hist_equalization",
    "percentile_stretch_1_99",
    "gamma_0_9",
    "adaptive_noise_3way",
    "clahe_rl_deblur_bilateral",
    # This segmentation block is the verified recovery path for tight,
    # two-line plates such as wrong_images/59DB05813.png.
    "component_mask_gray",
)

REQUESTED_TABLE_METADATA = {
    "adaptive_noise_3way": {
        "method_type": "Single-view",
        "benchmark_remark": "Adaptive preprocessing",
    },
    "clahe_rl_deblur_bilateral": {
        "method_type": "Single-view",
        "benchmark_remark": "Best among fixed pipelines",
    },
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else default
    except (TypeError, ValueError):
        return default


def _optional_number(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except (TypeError, ValueError):
        return None


def _display_name(name: str) -> str:
    if name == "adaptive_noise_3way":
        return "Adaptive Noise Router"
    if name == "clahe_rl_deblur_bilateral":
        return "CLAHE + Richardson-Lucy + Bilateral"
    if name == "richardson_lucy_deblur":
        return "Richardson-Lucy Deblur"
    replacements = {
        "clahe": "CLAHE",
        "rl": "Richardson-Lucy",
        "rgb": "RGB",
        "cv": "CV",
        "nlm": "NLM",
        "fft": "FFT",
        "haar": "Haar",
        "3way": "3-Way",
        "2way": "2-Way",
    }
    return " ".join(replacements.get(part, part.capitalize()) for part in name.split("_"))


def _pipeline_steps(config: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if bool(config.get("grayscale", False)):
        steps.append(f"Grayscale ({config.get('gray_channel', 'luma')})")
    if bool(config.get("autocontrast", False)):
        steps.append("Global autocontrast")
    if bool(config.get("histogram_equalization", False)):
        steps.append("Global histogram equalization")
    if _number(config.get("percentile_high")) > 0:
        steps.append(
            f"Percentile stretch {config.get('percentile_low', 0):g}-{config.get('percentile_high', 100):g}"
        )
    if _number(config.get("clahe_clip_limit")) > 0:
        steps.append(f"CLAHE clip {config.get('clahe_clip_limit'):g}")
    if _number(config.get("gamma"), 1.0) != 1.0:
        steps.append(f"Gamma {config.get('gamma'):g}")
    for field, suffix in (
        ("illumination", ""),
        ("deblur", " restoration"),
        ("denoise", ""),
        ("frequency_filter", " frequency filter"),
    ):
        value = str(config.get(field, "none"))
        if value != "none":
            steps.append(f"{value.replace('_', ' ').title()}{suffix}")
    if _number(config.get("sharpen_alpha")) > 0:
        steps.append(f"{config.get('sharpen_method', 'unsharp').title()} sharpening")
    morphology = str(config.get("morphology", "none"))
    if morphology != "none":
        steps.append(f"{morphology.replace('_', ' ').title()} morphology")
    threshold = str(config.get("threshold", "none"))
    if threshold != "none":
        steps.append(f"{threshold.title()} threshold")
    character_isolation = str(config.get("character_isolation", "none"))
    if character_isolation != "none":
        steps.append(f"{character_isolation.replace('_', ' ').title()} character isolation")
    return steps or ["RGB normalization only"]


def _impact_reason(name: str, config: dict[str, Any]) -> str:
    if str(config.get("character_isolation", "none")) != "none":
        return "Character-component isolation suppresses plate borders, glare, and background texture while retaining OCR strokes."
    if "clahe" in name:
        return "Local contrast enhancement separates strokes from uneven plate backgrounds."
    if "wavelet" in name:
        return "Haar soft-thresholding suppresses noise while retaining localized character edges."
    if str(config.get("frequency_filter", "none")) != "none":
        return "Frequency-domain high-boost filtering emphasizes fine detail and softened stroke boundaries."
    if str(config.get("deblur", "none")) != "none":
        return "Model-based restoration attempts to recover character strokes softened by optical blur."
    if str(config.get("denoise", "none")) != "none":
        return "Noise suppression improves stroke consistency while preserving the plate layout."
    if str(config.get("illumination", "none")) != "none":
        return "Illumination normalization reduces glare, shadows, and slow background variation."
    if str(config.get("threshold", "none")) != "none":
        return "Binarization isolates likely character strokes from the plate background."
    return "Preserves character geometry while improving the signal presented to PARSeq."


def _benchmark_rows(available_configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for csv_path in (*BENCHMARK_FILES, *TEST_BENCHMARK_FILES):
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        is_test = csv_path in TEST_BENCHMARK_FILES
        for row in frame.to_dict(orient="records"):
            name = str(row.get("name") or row.get("config") or "").strip()
            if name not in available_configs:
                continue
            candidate = {
                **row,
                "split": "test" if is_test else str(row.get("split", "validation")),
                "benchmark_source": f"{csv_path.parent.name}/{csv_path.name}",
                "_test_priority": is_test,
            }
            current = best_rows.get(name)
            if current is None or bool(candidate["_test_priority"]) > bool(
                current.get("_test_priority", False)
            ) or (
                bool(candidate["_test_priority"]) == bool(current.get("_test_priority", False))
                and (
                _number(candidate.get("exact_acc")), _number(candidate.get("char_acc"))
                ) > (_number(current.get("exact_acc")), _number(current.get("char_acc")))
            ):
                best_rows[name] = candidate
    return best_rows


def load_method_catalog(
    available_configs: dict[str, Any],
    rl_method: dict[str, Any] | None = None,
    learned_methods: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return only independently reusable blocks for plus/drag composition."""
    best_rows = _benchmark_rows(available_configs)
    catalog: list[dict[str, Any]] = []
    for name in UI_METHOD_NAMES:
        if name not in available_configs:
            continue
        cfg = available_configs[name]
        cfg_dict = asdict(cfg)
        row = best_rows.get(name, {})
        exact = _optional_number(row.get("exact_acc"))
        benchmark_available = bool(row) and exact is not None
        catalog.append(
            {
                "rank": len(catalog) + 1,
                "name": name,
                "display_name": _display_name(name),
                "topic": str(row.get("course_topic") or cfg.course_topic),
                "description": str(row.get("description") or cfg.description),
                "impact_reason": _impact_reason(name, cfg_dict),
                "pipeline_steps": _pipeline_steps(cfg_dict),
                "exact_acc": exact,
                "char_acc": _optional_number(row.get("char_acc")),
                "delta_exact": _optional_number(row.get("delta_exact")),
                "images_per_second": _optional_number(row.get("images_per_second")),
                "samples": int(_number(row.get("samples"))),
                "is_baseline": False,
                "benchmark_available": benchmark_available,
                "benchmark_split": str(row.get("split", "validation")) if benchmark_available else None,
                "benchmark_source": str(row.get("benchmark_source", "runtime_unmeasured")),
                "kind": "preprocessing_block",
                "filter_group": "imp",
                "available": True,
                "unavailable_reason": None,
                "experimental": False,
                "composable": True,
                "exclusive": False,
                "comparison_eligible": True,
                **REQUESTED_TABLE_METADATA.get(name, {}),
            }
        )

    if rl_method is not None:
        available = bool(rl_method.get("available", False))
        samples = int(_number(rl_method.get("samples")))
        catalog.append(
            {
                "rank": len(catalog) + 1,
                "name": "rl_deblur_restore",
                "display_name": "PixelRL Reinforcement Agent",
                "topic": "Reinforcement learning / PixelRL + A2C",
                "description": "A learned per-pixel restoration agent applies five correction steps before PARSeq OCR.",
                "impact_reason": "The policy chooses a local restoration action for every pixel instead of using one fixed blur kernel.",
                "pipeline_steps": [
                    "Resize to the 32x128 grayscale agent canvas",
                    "PixelRL actor-critic: five greedy action-map steps",
                    "Convert restored signal back to RGB for PARSeq",
                ],
                "exact_acc": _optional_number(rl_method.get("exact_acc")),
                "char_acc": _optional_number(rl_method.get("char_acc")),
                "delta_exact": _optional_number(rl_method.get("delta_exact")),
                "images_per_second": _optional_number(rl_method.get("images_per_second")),
                "samples": samples,
                "is_baseline": False,
                "benchmark_available": samples > 0,
                "benchmark_split": "degraded evaluation" if samples else None,
                "benchmark_source": str(rl_method.get("benchmark_source", "rl_deblur_eval")),
                "kind": "rl_agent",
                "filter_group": "rl",
                "available": available,
                "unavailable_reason": rl_method.get("unavailable_reason"),
                "experimental": True,
                "composable": available,
                "exclusive": False,
                "comparison_eligible": available,
            }
        )

    catalog.append(
        {
            "rank": len(catalog) + 1,
            "name": RECOVERY_METHOD_NAME,
            "display_name": "Verified RL",
            "topic": "Seven-candidate selector",
            "description": "Human-verified recovery routes are evaluated as a compact seven-candidate ensemble.",
            "impact_reason": "Known geometry, segmentation, illumination, morphology, wavelet, and thresholding recoveries are evaluated without reading the filename or target label.",
            "pipeline_steps": [
                "Generate baseline plus six verified recovery candidates",
                "Decode all seven candidates in one PARSeq batch",
                "Filter implausible plate strings and select normalized confidence",
            ],
            "exact_acc": None,
            "char_acc": None,
            "delta_exact": None,
            "images_per_second": None,
            "samples": 0,
            "is_baseline": False,
            "benchmark_available": False,
            "benchmark_split": None,
            "benchmark_source": "wrong_images_controlled_recovery",
            "kind": "recovery_selector",
            "filter_group": "rl",
            "available": True,
            "unavailable_reason": None,
            "experimental": True,
            "experimental_label": "RECOVERY",
            "catalog_badge": "RECOVERY",
            "featured": True,
            "candidate_view_count": 7,
            "candidate_view_label": "baseline + 6 verified recovery routes",
            "composable": True,
            "exclusive": True,
            "comparison_eligible": False,
        }
    )

    learned_definitions = {
        "calibrated_candidate_selector": {
            "display_name": "Calibrated Candidate Selector",
            "topic": "Learned routing / calibrated 65-view selector",
            "description": "A locked pairwise ranker groups 65 multi-scale OCR views plus six verified image-error routes and selects a calibrated result.",
            "impact_reason": "Vote strength, calibrated confidence, view reliability, plate shape and image geometry are combined with a small label-free priority for verified error-recovery routes.",
            "pipeline_steps": [
                "Generate baseline plus 64 resolution-aware OCR views",
                "Always run all six verified image-error recovery routes",
                "Aggregate duplicate candidate strings",
                "Apply the locked calibrated pairwise selector",
                "Give recovery routes a small bonus; strongly prioritize a content-matched route",
            ],
            "candidate_view_count": 71,
            "candidate_view_label": "65 standard views + 6 image-error recovery routes",
            "kind": "learned_selector",
            "filter_group": "imp",
            "method_type": "Multi-view",
            "benchmark_remark": "Best overall performance",
            "experimental": False,
        },
        "tta_65_view_consensus": {
            "display_name": "65-view TTA Consensus",
            "topic": "Multi-view / locked test-time augmentation consensus",
            "description": "A locked consensus rule selects from baseline plus 64 resolution-aware OCR views and six always-on verified image-error routes.",
            "impact_reason": "Agreement across zoom, upscale, full-plate and two-line-unwrapped views reduces dependence on one route, while verified recovery candidates receive a small selection bonus.",
            "pipeline_steps": [
                "Generate baseline plus 64 resolution-aware OCR views",
                "Always run all six verified image-error recovery routes",
                "Aggregate duplicate OCR strings and supporting views",
                "Apply consensus with a small recovery-route priority",
            ],
            "candidate_view_count": 71,
            "candidate_view_label": "65 standard views + 6 image-error recovery routes",
            "kind": "multi_view_selector",
            "filter_group": "imp",
            "method_type": "Multi-view",
            "benchmark_remark": "Oracle near-optimal",
            "experimental": False,
        },
        "contextual_bandit": {
            "display_name": "Contextual Bandit",
            "topic": "Reinforcement learning / Phase 4 one-step router",
            "description": "A frozen offline reward router chooses one complete restoration action or safely keeps the baseline.",
            "impact_reason": "The policy adapts restoration to image quality and OCR uncertainty while a learned gain margin allows abstention.",
            "pipeline_steps": [
                "Extract image-quality and PARSeq uncertainty context",
                "Predict reward for 10 complete actions",
                "Run the selected action or retain baseline",
            ],
            "kind": "rl_selector",
            "experimental": True,
        },
        "two_stage_ppo": {
            "display_name": "Two-stage PPO",
            "topic": "Reinforcement learning / Phase 5 actor-critic PPO",
            "description": "A teacher-guided actor-critic makes an initial restoration choice and may revise it in a second decision.",
            "impact_reason": "The second stage can reconsider a weak first action using the intermediate OCR observation rather than committing immediately.",
            "pipeline_steps": [
                "Build baseline state and contextual-bandit teacher prior",
                "Choose the first restoration action",
                "Optionally revise the action after intermediate OCR",
            ],
            "kind": "rl_selector",
            "experimental": True,
        },
        "auto_candidate_ppo": {
            "display_name": "Auto Candidate PPO",
            "topic": "Reinforcement learning / automatic top-20 policy",
            "description": "The auto policy scores 20 complete compositional candidates and applies a teacher-guarded PPO selection.",
            "impact_reason": "It evaluates complete candidate views jointly, including consensus between OCR strings, and falls back to the training baseline when gain is unsafe.",
            "pipeline_steps": [
                "Create the locked top-20 candidate views",
                "Extract per-candidate visual and OCR consensus features",
                "Select with candidate-set PPO and teacher safety guard",
            ],
            "candidate_view_count": 20,
            "candidate_view_label": "20 complete compositional views",
            "kind": "rl_selector",
            "experimental": True,
        },
    }
    for name, definition in learned_definitions.items():
        runtime = (learned_methods or {}).get(name, {})
        available = bool(runtime.get("available", False))
        samples = int(_number(runtime.get("samples")))
        catalog.append(
            {
                "rank": len(catalog) + 1,
                "name": name,
                **definition,
                "exact_acc": _optional_number(runtime.get("exact_acc")),
                "char_acc": _optional_number(runtime.get("char_acc")),
                "delta_exact": _optional_number(runtime.get("delta_exact")),
                "images_per_second": None,
                "samples": samples,
                "is_baseline": False,
                "benchmark_available": samples > 0 and runtime.get("exact_acc") is not None,
                "benchmark_split": "locked test" if samples else None,
                "benchmark_source": str(runtime.get("benchmark_source", "external_rl_pipeline")),
                "filter_group": str(definition.get("filter_group", "rl")),
                "available": available,
                "unavailable_reason": runtime.get("unavailable_reason"),
                "composable": available,
                "exclusive": True,
                "comparison_eligible": False,
            }
        )

    # Measured blocks lead the catalog by validation exact match, with
    # character accuracy as a deterministic tie-breaker. Unmeasured blocks
    # remain available after them without receiving an invented score.
    catalog.sort(
        key=lambda item: (
            item.get("filter_group") != "imp",
            not bool(item["benchmark_available"]),
            -_number(item["exact_acc"]) if item["benchmark_available"] else 0.0,
            -_number(item["char_acc"]) if item["benchmark_available"] else 0.0,
            not bool(item["available"]),
            int(item["rank"]),
        )
    )
    for rank, item in enumerate(catalog, start=1):
        item["rank"] = rank
    return catalog
