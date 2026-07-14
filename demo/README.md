# PARSeq Preprocessing Lab

Web demo for automatically locating a license plate in a full scene, cropping it, and running the fine-tuned PARSeq checkpoint with the preprocessing methods that improved benchmark exact-match accuracy.

## Run

From the repository root:

```powershell
pip install -r requirements.txt
pip install -r demo/requirements.txt
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The default checkpoint is:

```text
outputs/refinement_finetune/best_official_parseq_anpr.pt
```

The demo also discovers the locally trained YOLO26 plate detector at:

```text
../runs/yolo26_anpr/plate_detect_archive_yolo26m/weights/best.pt
```

To use another checkpoint or device:

```powershell
$env:PARSEQ_CHECKPOINT = "D:\path\to\best_official_parseq_anpr.pt"
$env:PARSEQ_DEVICE = "cuda"
$env:PARSEQ_REFINE_ITERS = "2"
$env:PLATE_DETECTOR_CHECKPOINT = "D:\path\to\plate_detector.pt"
$env:PLATE_DETECTOR_CONFIDENCE = "0.25"
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Both models are loaded lazily on the first Detect or Compare request. Auto-locate plate is enabled by default. Disable it in the interface when the uploaded image is already a tightly controlled crop and you want to bypass YOLO. Compare detects the plate once, preprocesses the shared crop with every displayed method, and submits them to PARSeq in one inference batch.
