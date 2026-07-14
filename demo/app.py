"""FastAPI entrypoint for the PARSeq preprocessing laboratory demo."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .inference_engine import InferenceEngine, image_to_data_url
    from .method_catalog import load_method_catalog
except ImportError:
    from inference_engine import InferenceEngine, image_to_data_url
    from method_catalog import load_method_catalog


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="PARSeq Preprocessing Lab",
    description="Technical ANPR inference and preprocessing comparison interface.",
    version="1.0.0",
)
engine = InferenceEngine()
catalog = load_method_catalog(engine.available_configs)
catalog_by_name = {item["name"]: item for item in catalog}


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
    return {"status": "ready", "model": engine.status(), "methods": len(catalog)}


@app.get("/api/methods")
def methods() -> dict:
    return {
        "methods": catalog,
        "default_method": catalog[0]["name"] if catalog else "train_baseline",
        "model": engine.status(),
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    method: str = Form(...),
    auto_detect: bool = Form(True),
) -> dict:
    if method not in catalog_by_name:
        raise HTTPException(status_code=400, detail=f"Method is not available in the demo: {method}")
    image = await read_image(file)
    try:
        result = engine.detect(image, method, auto_detect=auto_detect)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "result": {**result, "benchmark": catalog_by_name[method]},
        "original_image": image_to_data_url(image),
        "model": engine.status(),
    }


@app.post("/api/compare")
async def compare(file: UploadFile = File(...), auto_detect: bool = Form(True)) -> dict:
    image = await read_image(file)
    method_names = [item["name"] for item in catalog]
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
