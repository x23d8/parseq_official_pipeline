"""Lazy-loaded PARSeq inference engine used by the web demo."""

from __future__ import annotations

import base64
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


DEFAULT_CHECKPOINT = ROOT / "outputs" / "refinement_finetune" / "best_official_parseq_anpr.pt"
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
        self.available_configs = {config.name: config for config in SWEEP_CONFIGS}
        self._model = None
        self._detector = None
        self._detector_error: str | None = None
        self._model_cfg: OfficialPARSeqANPRConfig | None = None
        self._checkpoint_meta: dict[str, Any] = {}
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
        }

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
        return crop, {
            "enabled": True,
            "detected": True,
            "fallback_to_original": False,
            "reason": "highest_confidence_detection",
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name,
            "bbox": crop_box,
            "source_size": [original_width, original_height],
            "crop_size": [crop.width, crop.height],
            "candidate_count": int(len(boxes)),
            "detection_ms": detection_ms,
            "annotated_image": image_to_data_url(annotated),
            "crop_image": image_to_data_url(crop),
        }

    def _tensorize(self, image: Image.Image, config_name: str) -> tuple[Image.Image, torch.Tensor, str, float]:
        cfg = get_preprocessing_config(config_name)
        runtime_name = config_name
        if cfg.adaptive_policy != "none":
            runtime_name = _adaptive_config_name(image, cfg.adaptive_policy)
        started = time.perf_counter()
        processed = preprocess_plate_image(image, cfg).convert("RGB")
        interpolation = INTERPOLATIONS.get(cfg.resize_interpolation, InterpolationMode.BICUBIC)
        target_size = tuple(self._model_cfg.img_size if self._model_cfg else (32, 128))
        resize = (
            T.Resize(target_size, interpolation=interpolation)
            if cfg.resize_mode == "stretch"
            else LetterboxResize(target_size, interpolation)
        )
        tensor = T.Compose(
            [
                resize,
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )(processed)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return processed, tensor, runtime_name, elapsed_ms

    def detect(self, image: Image.Image, config_name: str, auto_detect: bool = True) -> dict[str, Any]:
        if config_name not in self.available_configs:
            raise KeyError(f"Unknown preprocessing config: {config_name}")
        self.ensure_loaded()
        plate_image, detection = self._plate_input(image, auto_detect=auto_detect)
        processed, tensor, runtime_name, preprocessing_ms = self._tensorize(plate_image, config_name)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            predictions, confidences = greedy_decode(
                self._model,
                tensor.unsqueeze(0).to(self.device),
                max_length=self._model_cfg.max_label_length,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "method": config_name,
            "runtime_method": runtime_name,
            "prediction": predictions[0],
            "confidence": float(confidences[0].detach().cpu().item()),
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "total_ms": detection["detection_ms"] + preprocessing_ms + inference_ms,
            "processed_image": image_to_data_url(processed),
            "detection": detection,
        }

    def compare(self, image: Image.Image, config_names: Iterable[str], auto_detect: bool = True) -> dict[str, Any]:
        names = [name for name in config_names if name in self.available_configs]
        if not names:
            raise ValueError("No valid preprocessing methods were provided.")
        self.ensure_loaded()
        plate_image, detection = self._plate_input(image, auto_detect=auto_detect)
        prepared = []
        tensors = []
        for name in names:
            processed, tensor, runtime_name, preprocessing_ms = self._tensorize(plate_image, name)
            prepared.append((name, processed, runtime_name, preprocessing_ms))
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
        for (name, processed, runtime_name, preprocessing_ms), prediction, confidence in zip(
            prepared, predictions, confidences.detach().cpu().tolist()
        ):
            rows.append(
                {
                    "method": name,
                    "runtime_method": runtime_name,
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
