"""FastAPI entrypoint for the PARSeq preprocessing laboratory demo."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .inference_engine import MAX_PIPELINE_STEPS, InferenceEngine, image_to_data_url
    from .method_catalog import load_method_catalog
except ImportError:
    from inference_engine import MAX_PIPELINE_STEPS, InferenceEngine, image_to_data_url
    from method_catalog import load_method_catalog


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UI_METHOD_LIMIT = 26

app = FastAPI(
    title="PARSeq Preprocessing Lab",
    description="Technical ANPR inference and preprocessing comparison interface.",
    version="1.0.0",
)
engine = InferenceEngine()
catalog = load_method_catalog(
    engine.available_configs,
    engine.rl_method_info(),
    engine.learned_method_info(),
)[:UI_METHOD_LIMIT]
catalog_by_name = {item["name"]: item for item in catalog}


def resolve_pipeline(method: str | None, pipeline: str | None) -> list[str]:
    if pipeline:
        try:
            decoded = json.loads(pipeline)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Pipeline must be a JSON array of method names.") from exc
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise HTTPException(status_code=400, detail="Pipeline must be a JSON array of method names.")
        names = decoded
    elif method:
        names = [method]
    else:
        raise HTTPException(status_code=400, detail="Select at least one processing method.")
    missing = [name for name in names if name not in catalog_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"Method is not available in the demo: {', '.join(missing)}")
    try:
        return engine.normalize_pipeline(names)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def pipeline_benchmark(names: list[str]) -> dict:
    if len(names) == 1:
        return {**catalog_by_name[names[0]], "is_composition": False}
    return {
        "display_name": " → ".join(catalog_by_name[name]["display_name"] for name in names),
        "description": "Custom ordered composition. Each method consumes the previous method's output.",
        "benchmark_available": False,
        "is_composition": True,
        "exact_acc": None,
        "char_acc": None,
        "delta_exact": None,
        "samples": 0,
    }


async def read_image(upload: UploadFile) -> Image.Image:
    content_type = upload.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a valid image file.")
    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image = Image.open(BytesIO(payload))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file could not be decoded as an image.") from exc


@app.get("/api/health")
def health() -> dict:
    checkpoint_ready = engine.checkpoint_path.is_file()
    return {
        "status": "ready" if checkpoint_ready else "degraded",
        "model": engine.status(),
        "methods": len(catalog),
        "method_limit": UI_METHOD_LIMIT,
        "max_pipeline_steps": MAX_PIPELINE_STEPS,
    }


@app.get("/api/methods")
def methods() -> dict:
    return {
        "methods": catalog,
        "default_method": catalog[0]["name"] if catalog else "train_baseline",
        "max_pipeline_steps": MAX_PIPELINE_STEPS,
        "method_limit": UI_METHOD_LIMIT,
        "model": engine.status(),
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    method: str | None = Form(None),
    pipeline: str | None = Form(None),
    auto_detect: bool = Form(True),
) -> dict:
    names = resolve_pipeline(method, pipeline)
    image = await read_image(file)
    try:
        result = engine.detect(image, names, auto_detect=auto_detect)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "result": {
            **result,
            "benchmark": pipeline_benchmark(names),
            "pipeline_methods": [catalog_by_name[name] for name in names],
        },
        "original_image": image_to_data_url(image),
        "model": engine.status(),
    }


@app.post("/api/compare")
async def compare(file: UploadFile = File(...), auto_detect: bool = Form(True)) -> dict:
    image = await read_image(file)
    method_names = [
        item["name"]
        for item in catalog
        if item.get("available", True) and item.get("comparison_eligible", True)
    ]
    try:
        comparison = engine.compare(image, method_names, auto_detect=auto_detect)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    enriched = []
    for result in comparison["results"]:
        enriched.append({**result, "benchmark": catalog_by_name[result["method"]]})
    return {
        **comparison,
        "results": enriched,
        "original_image": image_to_data_url(image),
        "model": engine.status(),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
