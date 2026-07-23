# PARSeq Preprocessing Lab

Web demo for automatically locating a license plate in a full scene, cropping it, and running the fine-tuned PARSeq checkpoint with reusable image-processing blocks.

The UI exposes 20 standalone image-processing blocks, one verified hard-case
recovery ensemble, and five learned methods.
It does not list the
99 pre-baked benchmark combinations: build those combinations by clicking `+`
or dragging blocks into the ordered pipeline. Standalone blocks without a
matching historical validation run are labeled `N/A` instead of receiving an
invented score.

The Method Catalog initially shows the top 10 ranked blocks. **Show more**
opens the remaining blocks in a separate group; those methods retain the same
`+`, detail-selection, search, and drag-to-pipeline behavior.
The verified recovery ensemble is pinned in the initial group for hard-case demos.

Images can be selected from a local file, dropped onto the input area, pasted
with `Ctrl+V` / `Cmd+V`, or read with the **Paste image** button when the browser
grants clipboard permission.

The catalog exposes **Calibrated Candidate**, **Contextual Bandit**, **Two-stage
PPO**, **PixelRL/A2C**, and **Auto Candidate PPO** when `RL_PIPELINE_ROOT`
points to `D:\NEO\rl_pipeline`. The local checkout already has junctions for
the shared RL source folders; checkpoints remain in the RL project and are not
copied. Add methods with `+` or drag them into the pipeline
composer. Chips execute from left to right and can be reordered by drag-and-drop
or their arrow controls. Custom compositions are marked exploratory because
they do not inherit benchmark accuracy from individual methods.

Calibrated Candidate and the three selector policies are complete OCR
orchestrators and therefore run alone. PixelRL is an image-restoration block
and remains composable. The `IMP` / `RL methods` checkboxes filter the catalog,
and the top-bar theme button persists light/dark mode locally.

The UI spells out **Richardson–Lucy** for classical deconvolution filters; the
`RL AGENT` badge is reserved for reinforcement learning.

## Run

From the repository root:

```powershell
pip install -r requirements.txt
pip install -r demo/requirements.txt
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Run anywhere with Docker

The container includes the PARSeq OCR checkpoint and plate detector, runs on CPU
by default, and accepts the standard `PORT` environment variable.

```powershell
docker compose up --build
```

Open `http://localhost:7860`. The equivalent plain Docker commands are:

```powershell
docker build -t parseq-anpr-demo .
docker run --rm -p 7860:7860 parseq-anpr-demo
```

To include the external RL methods with plain Docker, mount the sibling project:

```powershell
docker run --rm -p 7860:7860 -e RL_PIPELINE_ROOT=/opt/rl_pipeline `
  -v D:\NEO\rl_pipeline:/opt/rl_pipeline:ro parseq-anpr-demo
```

For a GPU host, install the NVIDIA container runtime, pass `--gpus all`, and set
`PARSEQ_DEVICE=cuda`. The supplied image contains CPU-only PyTorch; build a CUDA
variant before enabling that setting.

## Deploy to Hugging Face Spaces

If your account can create dynamic Spaces, create a new **Docker Space**, then
push this repository to the Space repository. The YAML metadata at the top of
the root `README.md` selects Docker and port `7860`; no environment variables
are required. CPU Basic has enough memory for the demo, but the first recognition
request will be slower while the models load. As of July 2026, Hugging Face may
require a PRO subscription before an account can create a new Docker or Gradio
Space.

```powershell
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space HEAD:main
```

The same Dockerfile can be deployed to any service that accepts a container.
Expose its `PORT` value and route health checks to `/api/health`.

## Temporary public preview

For a review link from a development machine, start the app and use a Cloudflare
Quick Tunnel:

```powershell
python -m demo
cloudflared tunnel --url http://127.0.0.1:7860
```

Quick Tunnel URLs are public and last only while both processes remain running;
they are intended for demos and development, not production uptime.

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
$env:RL_PIPELINE_ROOT = "D:\path\to\rl_pipeline"
# Optional: $env:RL_DEBLUR_CHECKPOINT = "D:\path\to\best_deblur_agent.pt"
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

PARSeq, YOLO, and the RL agent are loaded lazily. Auto-locate plate is enabled
by default. When YOLO finds that the plate already occupies most of the upload,
the original tight crop is preserved so boundary characters are not clipped. Compare
detects once and evaluates each available method independently in one PARSeq
batch; it deliberately does not enumerate custom method combinations.
