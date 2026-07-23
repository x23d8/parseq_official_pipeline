"""Lazy-loaded PARSeq inference engine used by the web demo."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

try:
    from .rl_runtime import LearnedSelectorRuntime, SELECTOR_METHODS
    from .recovery_runtime import (
        RECOVERY_METHOD_NAME,
        RECOVERY_VIEWS,
        is_plausible_vietnamese_plate,
        normalized_confidence,
        select_recovery_candidate,
    )
except ImportError:
    from rl_runtime import LearnedSelectorRuntime, SELECTOR_METHODS
    from recovery_runtime import (
        RECOVERY_METHOD_NAME,
        RECOVERY_VIEWS,
        is_plausible_vietnamese_plate,
        normalized_confidence,
        select_recovery_candidate,
    )


ROOT = Path(__file__).resolve().parents[1]
PARSEQ_DIR = ROOT / "parseq"
PREPROCESSING_DIR = ROOT / "preprocessing_best_config"
TRAINING_DIR = ROOT / "train_no_refinement"
for path in (PARSEQ_DIR, PREPROCESSING_DIR, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parseq_official_anpr_pipeline import (  # noqa: E402
    OfficialPARSeqANPRConfig,
    create_official_parseq_model,
    greedy_decode,
    set_decode_mode,
)
from preprocessing import (  # noqa: E402
    SWEEP_CONFIGS,
    _adaptive_config_name,
    get_preprocessing_config,
    preprocess_plate_image,
)


DEFAULT_CHECKPOINT = ROOT / "outputs" / "refinement_finetune_20260721_144512" / "best_official_parseq_anpr.pt"
DEFAULT_RL_PIPELINE_CANDIDATES = (
    ROOT / "rl_pipeline",
    ROOT.parent / "rl_pipeline",
    ROOT.parents[1] / "rl_pipeline",
)
RL_METHOD_NAME = "rl_deblur_restore"
MAX_PIPELINE_STEPS = 5
DEFAULT_DETECTOR_CANDIDATES = (
    ROOT / "weights" / "plate_detector.pt",
    ROOT.parent / "runs" / "yolo26_anpr" / "plate_detect_archive_yolo26m" / "weights" / "best.pt",
)
INTERPOLATIONS = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}
TIGHT_CROP_MIN_DETECTION_COVERAGE = 0.70


class LetterboxResize:
    def __init__(self, size: tuple[int, int], interpolation: InterpolationMode):
        self.height, self.width = int(size[0]), int(size[1])
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        ratio = min(self.width / max(image.width, 1), self.height / max(image.height, 1))
        resized = T.functional.resize(
            image,
            [max(1, round(image.height * ratio)), max(1, round(image.width * ratio))],
            interpolation=self.interpolation,
        )
        canvas = Image.new("RGB", (self.width, self.height), (127, 127, 127))
        canvas.paste(resized, ((self.width - resized.width) // 2, (self.height - resized.height) // 2))
        return canvas


def image_to_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class InferenceEngine:
    def __init__(self) -> None:
        configured = os.environ.get("PARSEQ_CHECKPOINT", "").strip()
        self.checkpoint_path = Path(configured).expanduser().resolve() if configured else DEFAULT_CHECKPOINT
        requested_device = os.environ.get("PARSEQ_DEVICE", "auto").strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.refine_iters = int(os.environ.get("PARSEQ_REFINE_ITERS", "2"))
        configured_detector = os.environ.get("PLATE_DETECTOR_CHECKPOINT", "").strip()
        if configured_detector:
            self.detector_checkpoint_path = Path(configured_detector).expanduser().resolve()
        else:
            self.detector_checkpoint_path = next(
                (path for path in DEFAULT_DETECTOR_CANDIDATES if path.exists()),
                DEFAULT_DETECTOR_CANDIDATES[0],
            )
        self.detector_confidence = float(os.environ.get("PLATE_DETECTOR_CONFIDENCE", "0.25"))
        self.detector_margin = float(os.environ.get("PLATE_DETECTOR_MARGIN", "0.04"))
        configured_rl_root = os.environ.get("RL_PIPELINE_ROOT", "").strip()
        self.rl_pipeline_root = (
            Path(configured_rl_root).expanduser().resolve()
            if configured_rl_root
            else next(
                (path.resolve() for path in DEFAULT_RL_PIPELINE_CANDIDATES if path.is_dir()),
                DEFAULT_RL_PIPELINE_CANDIDATES[0],
            )
        )
        configured_rl_checkpoint = os.environ.get("RL_DEBLUR_CHECKPOINT", "").strip()
        self.rl_checkpoint_path = (
            Path(configured_rl_checkpoint).expanduser().resolve()
            if configured_rl_checkpoint
            else self.rl_pipeline_root / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"
        )
        self.rl_source_path = self.rl_pipeline_root / "parseq_rl_deblur_data"
        self.available_configs = {config.name: config for config in SWEEP_CONFIGS}
        self._model = None
        self._detector = None
        self._detector_error: str | None = None
        self._model_cfg: OfficialPARSeqANPRConfig | None = None
        self._checkpoint_meta: dict[str, Any] = {}
        self._rl_method = None
        self._rl_canvas = None
        self._rl_error: str | None = None
        self._selector_runtime = LearnedSelectorRuntime(self.rl_pipeline_root, self.device)
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "refine_iters": self.refine_iters,
            "epoch": self._checkpoint_meta.get("epoch"),
            "detector": {
                "loaded": self._detector is not None,
                "checkpoint": str(self.detector_checkpoint_path),
                "checkpoint_exists": self.detector_checkpoint_path.exists(),
                "confidence_threshold": self.detector_confidence,
                "error": self._detector_error,
            },
            "rl": {
                "loaded": self._rl_method is not None,
                "root": str(self.rl_pipeline_root),
                "checkpoint": str(self.rl_checkpoint_path),
                "checkpoint_exists": self.rl_checkpoint_path.exists(),
                "source_exists": self.rl_source_path.is_dir(),
                "error": self._rl_error,
                "selectors": self._selector_runtime.availability(),
            },
        }

    def learned_method_info(self) -> dict[str, dict[str, Any]]:
        """Return availability plus measured metadata for external learned methods."""
        info = self._selector_runtime.availability()
        benchmark_specs = {
            "calibrated_candidate_selector": (
                self.rl_pipeline_root / "reinforcement_learning/phase_2_calibrated_selector/results/summary.json",
                ("test", "phase2"),
                "reinforcement_learning/phase_2_calibrated_selector/results/summary.json",
            ),
            "contextual_bandit": (
                self.rl_pipeline_root / "reinforcement_learning/common_rl_benchmark/summary.json",
                ("results", "test", "contextual_bandit_phase4"),
                "reinforcement_learning/common_rl_benchmark/summary.json",
            ),
            "two_stage_ppo": (
                self.rl_pipeline_root / "reinforcement_learning/common_rl_benchmark/summary.json",
                ("results", "test", "ppo_phase5"),
                "reinforcement_learning/common_rl_benchmark/summary.json",
            ),
            "auto_candidate_ppo": (
                self.rl_pipeline_root / "reinforcement_learning/auto/results/experiment/experiment_summary.json",
                ("comparisons", "test", "with_auto_rl"),
                "reinforcement_learning/auto/results/experiment/experiment_summary.json",
            ),
        }
        for name, (summary_path, keys, source) in benchmark_specs.items():
            item = info[name]
            item.update(samples=0, exact_acc=None, char_acc=None, delta_exact=None, benchmark_source=source)
            try:
                summary_payload: Any = json.loads(summary_path.read_text(encoding="utf-8"))
                payload: Any = summary_payload
                for key in keys:
                    payload = payload[key]
                item.update(
                    samples=int(payload.get("samples", 0)),
                    exact_acc=float(payload["exact_acc"]),
                    char_acc=float(payload["char_acc"]),
                )
                if name == "calibrated_candidate_selector":
                    baseline = summary_payload["test"]["baseline"]
                elif name in {"contextual_bandit", "two_stage_ppo"}:
                    baseline = summary_payload["results"]["test"]["raw_parseq"]
                else:
                    baseline = summary_payload["comparisons"]["test"]["without_rl"]
                item["delta_exact"] = float(payload["exact_acc"]) - float(baseline["exact_acc"])
            except (OSError, KeyError, TypeError, ValueError):
                pass
        return info

    def rl_method_info(self) -> dict[str, Any]:
        available = self.rl_checkpoint_path.is_file() and self.rl_source_path.is_dir()
        info: dict[str, Any] = {
            "available": available,
            "unavailable_reason": None,
            "samples": 0,
            "exact_acc": 0.0,
            "char_acc": 0.0,
            "delta_exact": 0.0,
            "images_per_second": 0.0,
            "benchmark_source": "outputs/rl_deblur/eval_summary.json",
        }
        if not available:
            missing = []
            if not self.rl_checkpoint_path.is_file():
                missing.append(f"checkpoint {self.rl_checkpoint_path}")
            if not self.rl_source_path.is_dir():
                missing.append(f"source {self.rl_source_path}")
            info["unavailable_reason"] = "Missing " + " and ".join(missing)
            return info
        summary_path = self.rl_pipeline_root / "outputs" / "rl_deblur" / "eval_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            exact = summary.get("exact_acc", {})
            cer = summary.get("cer", {})
            info.update(
                samples=int(summary.get("num_samples", 0)),
                exact_acc=float(exact.get("rl", 0.0)),
                char_acc=1.0 - float(cer.get("rl", 1.0)),
                delta_exact=float(exact.get("rl", 0.0)) - float(exact.get("blurred", 0.0)),
            )
        except (OSError, ValueError, TypeError):
            # The checkpoint remains usable even when its optional benchmark summary is unavailable.
            pass
        return info

    def is_method_available(self, name: str) -> bool:
        if name == RECOVERY_METHOD_NAME:
            return True
        if name == RL_METHOD_NAME:
            return bool(self.rl_method_info()["available"])
        if name in SELECTOR_METHODS:
            return bool(self.learned_method_info()[name]["available"])
        return name in self.available_configs

    def normalize_pipeline(self, method_names: Iterable[str]) -> list[str]:
        names = [str(name).strip() for name in method_names if str(name).strip()]
        if not names:
            raise ValueError("The processing pipeline must contain at least one method.")
        if len(names) > MAX_PIPELINE_STEPS:
            raise ValueError(f"A processing pipeline can contain at most {MAX_PIPELINE_STEPS} methods.")
        if len(set(names)) != len(names):
            raise ValueError("A processing method cannot appear more than once in the same pipeline.")
        known_special = {RL_METHOD_NAME, RECOVERY_METHOD_NAME, *SELECTOR_METHODS}
        unknown = [name for name in names if name not in self.available_configs and name not in known_special]
        if unknown:
            raise KeyError(f"Unknown preprocessing method(s): {', '.join(unknown)}")
        selected_orchestrators = [
            name for name in names if name in SELECTOR_METHODS or name == RECOVERY_METHOD_NAME
        ]
        if selected_orchestrators and len(names) != 1:
            raise ValueError("Complete selector methods must be used alone.")
        unavailable = [name for name in names if not self.is_method_available(name)]
        if unavailable:
            raise RuntimeError(f"Unavailable processing method(s): {', '.join(unavailable)}")
        return names

    def _load_checkpoint(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. Set PARSEQ_CHECKPOINT to a valid .pt file."
            )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        saved = checkpoint.get("config", {})
        allowed = {field.name for field in fields(OfficialPARSeqANPRConfig)}
        cfg_values = {key: value for key, value in saved.items() if key in allowed}
        if "img_size" in cfg_values:
            cfg_values["img_size"] = tuple(cfg_values["img_size"])
        cfg_values.update(pretrained=False, refine_iters=self.refine_iters, augment=False, decode_ar=True)
        model_cfg = OfficialPARSeqANPRConfig(**cfg_values)
        model = create_official_parseq_model(model_cfg, device=self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        set_decode_mode(model, refine_iters=self.refine_iters, decode_ar=True)
        model.eval()
        self._model = model
        self._model_cfg = model_cfg
        self._checkpoint_meta = {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
            "architecture": checkpoint.get("architecture", "official_strhub_parseq"),
        }

    def ensure_loaded(self) -> None:
        with self._lock:
            self._load_checkpoint()

    def _load_detector(self) -> None:
        if self._detector is not None or self._detector_error is not None:
            return
        if not self.detector_checkpoint_path.exists():
            self._detector_error = (
                f"Detector checkpoint not found: {self.detector_checkpoint_path}. "
                "Set PLATE_DETECTOR_CHECKPOINT or disable auto plate detection."
            )
            return
        try:
            # Ultralytics writes a small settings file on first import. Keep it outside the repository.
            yolo_config_dir = Path(tempfile.gettempdir()) / "parseq_demo_yolo"
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
            from ultralytics import YOLO

            self._detector = YOLO(str(self.detector_checkpoint_path), task="detect")
        except Exception as exc:  # Detector failure must not disable manual cropped-image OCR.
            self._detector_error = f"Could not load the plate detector: {exc}"

    @staticmethod
    def _is_tight_plate_input(
        source_size: tuple[int, int], crop_box: list[int],
    ) -> bool:
        """Keep uploads that the detector says are already mostly plate pixels.

        Re-cropping such images has little upside and can remove a boundary
        character when a detector box ends just inside the plate border.
        """
        source_width, source_height = source_size
        source_area = max(source_width * source_height, 1)
        crop_width = max(crop_box[2] - crop_box[0], 0)
        crop_height = max(crop_box[3] - crop_box[1], 0)
        return (crop_width * crop_height) / source_area >= TIGHT_CROP_MIN_DETECTION_COVERAGE

    def _plate_input(self, image: Image.Image, auto_detect: bool) -> tuple[Image.Image, dict[str, Any]]:
        image = image.convert("RGB")
        original_width, original_height = image.size
        if not auto_detect:
            return image, {
                "enabled": False,
                "detected": False,
                "fallback_to_original": True,
                "reason": "manual_crop_mode",
                "confidence": None,
                "class_id": None,
                "class_name": None,
                "bbox": [0, 0, original_width, original_height],
                "source_size": [original_width, original_height],
                "crop_size": [original_width, original_height],
                "detection_ms": 0.0,
                "annotated_image": image_to_data_url(image),
                "crop_image": image_to_data_url(image),
            }

        with self._lock:
            self._load_detector()
        if self._detector is None:
            return image, {
                "enabled": True,
                "detected": False,
                "fallback_to_original": True,
                "reason": self._detector_error or "detector_unavailable",
                "confidence": None,
                "class_id": None,
                "class_name": None,
                "bbox": [0, 0, original_width, original_height],
                "source_size": [original_width, original_height],
                "crop_size": [original_width, original_height],
                "detection_ms": 0.0,
                "annotated_image": image_to_data_url(image),
                "crop_image": image_to_data_url(image),
            }

        detector_device: str | int = "cpu"
        if self.device.type == "cuda":
            detector_device = int(self.device.index or 0)
        started = time.perf_counter()
        with self._lock:
            predictions = self._detector.predict(
                source=image,
                imgsz=640,
                conf=self.detector_confidence,
                iou=0.7,
                max_det=10,
                device=detector_device,
                verbose=False,
            )
        detection_ms = (time.perf_counter() - started) * 1000.0
        boxes = predictions[0].boxes if predictions else None
        if boxes is None or len(boxes) == 0:
            return image, {
                "enabled": True,
                "detected": False,
                "fallback_to_original": True,
                "reason": "no_plate_above_threshold",
                "confidence": None,
                "class_id": None,
                "class_name": None,
                "bbox": [0, 0, original_width, original_height],
                "source_size": [original_width, original_height],
                "crop_size": [original_width, original_height],
                "detection_ms": detection_ms,
                "annotated_image": image_to_data_url(image),
                "crop_image": image_to_data_url(image),
            }

        confidences = boxes.conf.detach().cpu()
        best_index = int(torch.argmax(confidences).item())
        best_box = boxes[best_index]
        raw_box = [float(value) for value in best_box.xyxy[0].detach().cpu().tolist()]
        confidence = float(best_box.conf[0].detach().cpu().item())
        class_id = int(best_box.cls[0].detach().cpu().item())
        names = getattr(self._detector, "names", {})
        class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else class_id)

        left, top, right, bottom = raw_box
        margin_x = max((right - left) * self.detector_margin, 2.0)
        margin_y = max((bottom - top) * self.detector_margin, 2.0)
        crop_box = [
            max(0, int(left - margin_x)),
            max(0, int(top - margin_y)),
            min(original_width, int(right + margin_x + 0.999)),
            min(original_height, int(bottom + margin_y + 0.999)),
        ]
        crop = image.crop(tuple(crop_box)).convert("RGB")

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(min(original_width, original_height) / 180))
        draw.rectangle(crop_box, outline=(36, 87, 255), width=line_width)
        label = f"PLATE {confidence * 100:.1f}% / {class_name}"
        text_box = draw.textbbox((crop_box[0], crop_box[1]), label)
        text_height = text_box[3] - text_box[1] + 8
        label_top = max(0, crop_box[1] - text_height)
        label_right = min(original_width, crop_box[0] + text_box[2] - text_box[0] + 10)
        draw.rectangle((crop_box[0], label_top, label_right, crop_box[1]), fill=(36, 87, 255))
        draw.text((crop_box[0] + 5, label_top + 3), label, fill=(255, 255, 255))

        tight_input = self._is_tight_plate_input(
            (original_width, original_height), crop_box
        )
        ocr_crop = image if tight_input else crop
        return ocr_crop, {
            "enabled": True,
            "detected": True,
            "fallback_to_original": tight_input,
            "reason": "tight_plate_input_preserved" if tight_input else "highest_confidence_detection",
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name,
            "bbox": crop_box,
            "source_size": [original_width, original_height],
            "crop_size": [ocr_crop.width, ocr_crop.height],
            "detector_crop_size": [crop.width, crop.height],
            "candidate_count": int(len(boxes)),
            "detection_ms": detection_ms,
            "annotated_image": image_to_data_url(annotated),
            "crop_image": image_to_data_url(ocr_crop),
        }

    def _load_rl_method(self) -> None:
        if self._rl_method is not None:
            return
        if self._rl_error is not None:
            raise RuntimeError(self._rl_error)
        if not self.rl_source_path.is_dir() or not self.rl_checkpoint_path.is_file():
            self._rl_error = str(self.rl_method_info().get("unavailable_reason") or "RL deblur unavailable")
            raise RuntimeError(self._rl_error)
        try:
            if str(self.rl_source_path) not in sys.path:
                sys.path.insert(0, str(self.rl_source_path))
            from image_processing_study.methods import to_canvas_gray, try_build_rl_deblur_method

            method = try_build_rl_deblur_method(self.rl_checkpoint_path, device=str(self.device))
            if method is None:
                raise RuntimeError("The RL deblur checkpoint could not be loaded.")
            self._rl_method = method
            self._rl_canvas = to_canvas_gray
        except Exception as exc:
            self._rl_error = f"Could not load RL deblur agent: {exc}"
            raise RuntimeError(self._rl_error) from exc

    def _apply_method(self, image: Image.Image, name: str) -> tuple[Image.Image, str, float]:
        started = time.perf_counter()
        if name == RL_METHOD_NAME:
            with self._lock:
                self._load_rl_method()
                canvas = self._rl_canvas(image, self._rl_method.resample)
                restored = self._rl_method.process(canvas)
            processed = Image.fromarray(restored).convert("RGB")
            runtime_name = RL_METHOD_NAME
        else:
            cfg = get_preprocessing_config(name)
            runtime_name = name
            if cfg.adaptive_policy != "none":
                runtime_name = _adaptive_config_name(image, cfg.adaptive_policy)
            processed = preprocess_plate_image(image, cfg).convert("RGB")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return processed, runtime_name, elapsed_ms

    def _tensor_from_processed(self, image: Image.Image, terminal_method: str) -> torch.Tensor:
        cfg = get_preprocessing_config(terminal_method) if terminal_method in self.available_configs else None
        interpolation = INTERPOLATIONS.get(
            cfg.resize_interpolation if cfg is not None else "bicubic",
            InterpolationMode.BICUBIC,
        )
        target_size = tuple(self._model_cfg.img_size if self._model_cfg else (32, 128))
        resize = (
            T.Resize(target_size, interpolation=interpolation)
            if cfg is None or cfg.resize_mode == "stretch"
            else LetterboxResize(target_size, interpolation)
        )
        return T.Compose(
            [
                resize,
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )(image)

    def _prepare_pipeline(
        self,
        image: Image.Image,
        method_names: Iterable[str],
    ) -> tuple[Image.Image, torch.Tensor, list[str], list[dict[str, Any]], float]:
        names = self.normalize_pipeline(method_names)
        processed = image.convert("RGB")
        runtime_names: list[str] = []
        timings: list[dict[str, Any]] = []
        for index, name in enumerate(names, start=1):
            processed, runtime_name, elapsed_ms = self._apply_method(processed, name)
            runtime_names.append(runtime_name)
            timings.append(
                {
                    "position": index,
                    "method": name,
                    "runtime_method": runtime_name,
                    "milliseconds": elapsed_ms,
                }
            )
        tensor_started = time.perf_counter()
        tensor = self._tensor_from_processed(processed, names[-1])
        tensor_ms = (time.perf_counter() - tensor_started) * 1000.0
        return processed, tensor, runtime_names, timings, sum(item["milliseconds"] for item in timings) + tensor_ms

    def _run_recovery_ensemble(self, image: Image.Image) -> dict[str, Any]:
        preprocessing_started = time.perf_counter()
        prepared: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        for view in RECOVERY_VIEWS:
            transformed = view.apply_geometry(image)
            processed, tensor, runtime_names, timings, _elapsed = self._prepare_pipeline(
                transformed, view.pipeline
            )
            prepared.append(
                {
                    "view": view,
                    "processed": processed,
                    "runtime_names": runtime_names,
                    "timings": timings,
                }
            )
            tensors.append(tensor)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0

        inference_started = time.perf_counter()
        with self._lock, torch.inference_mode():
            predictions, confidences = greedy_decode(
                self._model,
                torch.stack(tensors).to(self.device),
                max_length=self._model_cfg.max_label_length,
            )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        candidates: list[dict[str, Any]] = []
        for index, (item, prediction, confidence) in enumerate(
            zip(prepared, predictions, confidences.detach().cpu().tolist())
        ):
            candidates.append(
                {
                    "index": index,
                    "view": item["view"].name,
                    "pipeline": list(item["view"].pipeline),
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "normalized_confidence": normalized_confidence(confidence, prediction),
                    "plausible_plate": is_plausible_vietnamese_plate(prediction),
                }
            )
        selection_started = time.perf_counter()
        selected = select_recovery_candidate(candidates)
        selection_ms = (time.perf_counter() - selection_started) * 1000.0
        selected_item = prepared[int(selected["index"])]
        selected_view = selected_item["view"]
        step_timings = [
            {
                "position": 1,
                "method": RECOVERY_METHOD_NAME,
                "runtime_method": f"Generate {len(RECOVERY_VIEWS)} verified candidates",
                "milliseconds": preprocessing_ms + inference_ms,
            },
            {
                "position": 2,
                "method": selected_view.name,
                "runtime_method": "Selected: " + " -> ".join(selected_view.pipeline),
                "milliseconds": selection_ms,
            },
        ]
        return {
            "method": RECOVERY_METHOD_NAME,
            "pipeline": [RECOVERY_METHOD_NAME],
            "runtime_pipeline": [
                RECOVERY_METHOD_NAME,
                selected_view.name,
                *selected_item["runtime_names"],
            ],
            "runtime_method": " -> ".join(selected_view.pipeline),
            "step_timings": step_timings,
            "prediction": selected["prediction"],
            "confidence": selected["confidence"],
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "processed_image": image_to_data_url(selected_item["processed"]),
            "selector_trace": {
                "algorithm": "format_guarded_normalized_confidence",
                "candidate_count": len(candidates),
                "plausible_candidate_count": sum(
                    int(row["plausible_plate"]) for row in candidates
                ),
                "selected_view": selected_view.name,
                "selected_pipeline": list(selected_view.pipeline),
                "normalized_confidence": selected["normalized_confidence"],
                "candidates": candidates,
            },
        }

    def detect(
        self,
        image: Image.Image,
        config_name: str | Iterable[str],
        auto_detect: bool = True,
    ) -> dict[str, Any]:
        names = self.normalize_pipeline([config_name] if isinstance(config_name, str) else config_name)
        self.ensure_loaded()
        plate_image, detection = self._plate_input(image, auto_detect=auto_detect)
        if len(names) == 1 and names[0] == RECOVERY_METHOD_NAME:
            result = self._run_recovery_ensemble(plate_image)
            result["detection"] = detection
            result["total_ms"] = (
                detection["detection_ms"] + result["preprocessing_ms"] + result["inference_ms"]
            )
            return result
        if len(names) == 1 and names[0] in SELECTOR_METHODS:
            result = self._selector_runtime.run(self, plate_image, names[0])
            processed = result.pop("processed_image_pil")
            result["processed_image"] = image_to_data_url(processed)
            result["detection"] = detection
            result["total_ms"] = detection["detection_ms"] + result["preprocessing_ms"] + result["inference_ms"]
            return result
        processed, tensor, runtime_names, step_timings, preprocessing_ms = self._prepare_pipeline(
            plate_image, names
        )
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            predictions, confidences = greedy_decode(
                self._model,
                tensor.unsqueeze(0).to(self.device),
                max_length=self._model_cfg.max_label_length,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "method": names[0] if len(names) == 1 else "pipeline:" + "|".join(names),
            "pipeline": names,
            "runtime_pipeline": runtime_names,
            "runtime_method": " -> ".join(runtime_names),
            "step_timings": step_timings,
            "prediction": predictions[0],
            "confidence": float(confidences[0].detach().cpu().item()),
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "total_ms": detection["detection_ms"] + preprocessing_ms + inference_ms,
            "processed_image": image_to_data_url(processed),
            "detection": detection,
        }

    def compare(self, image: Image.Image, config_names: Iterable[str], auto_detect: bool = True) -> dict[str, Any]:
        names = [name for name in config_names if self.is_method_available(name)]
        if not names:
            raise ValueError("No valid preprocessing methods were provided.")
        self.ensure_loaded()
        plate_image, detection = self._plate_input(image, auto_detect=auto_detect)
        prepared = []
        tensors = []
        for name in names:
            processed, tensor, runtime_names, step_timings, preprocessing_ms = self._prepare_pipeline(
                plate_image, [name]
            )
            prepared.append((name, processed, runtime_names, step_timings, preprocessing_ms))
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(self.device)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            predictions, confidences = greedy_decode(
                self._model,
                batch,
                max_length=self._model_cfg.max_label_length,
            )
        model_ms = (time.perf_counter() - started) * 1000.0
        rows = []
        for (name, processed, runtime_names, step_timings, preprocessing_ms), prediction, confidence in zip(
            prepared, predictions, confidences.detach().cpu().tolist()
        ):
            rows.append(
                {
                    "method": name,
                    "pipeline": [name],
                    "runtime_pipeline": runtime_names,
                    "runtime_method": " -> ".join(runtime_names),
                    "step_timings": step_timings,
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "preprocessing_ms": preprocessing_ms,
                    "processed_image": image_to_data_url(processed),
                }
            )
        return {
            "results": rows,
            "model_batch_ms": model_ms,
            "method_count": len(rows),
            "detection": detection,
        }
