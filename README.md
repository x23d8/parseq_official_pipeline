---
title: PARSeq Vietnamese ANPR Demo
emoji: 🚘
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
fullWidth: true
short_description: Detect, restore and recognize Vietnamese license plates with PARSeq and PixelRL.
---

# PARSeq ANPR: Calibrated Candidate Selector + PixelRL

Đây là repo của **calibrated candidate selector**, đồng thời là điểm chạy demo tích hợp cho hệ thống nhận dạng biển số gồm:

- YOLO phát hiện và cắt biển số từ ảnh cảnh.
- Các pipeline xử lý ảnh tạo nhiều candidate.
- PARSeq nhận dạng từng candidate.
- Calibrated candidate selector chọn kết quả OCR đáng tin cậy.
- PixelRL/A2C phục hồi ảnh theo từng pixel trước khi đưa vào PARSeq.

Hệ thống đầy đủ được chia thành hai repo:

1. [x23d8/calibrated-candidate-selector-for-PARSeq](https://github.com/x23d8/calibrated-candidate-selector-for-PARSeq) — PARSeq, fine-tuning, 65-view inference, calibrated selector, detector và web demo.
2. [x23d8/PixelRL-PARSeq](https://github.com/x23d8/PixelRL-PARSeq) — PixelRL/A2C, các policy RL, trajectory cache và các artifact phục vụ demo.

## Quan hệ dependency giữa hai repo

Hai repo là **dependency của nhau ở mức mã nguồn, checkpoint và artifact**, không phải circular dependency giữa hai package trên PyPI.

```text
calibrated-candidate-selector-for-PARSeq
    ├── cung cấp PARSeq source + OCR checkpoint + manifest/dataset
    │                              │
    │                              ▼
    │                    PixelRL-PARSeq
    │                 train/evaluate PixelRL và policy RL
    │                              │
    └──── web demo nạp source + checkpoint RL ◄────┘
                  qua RL_PIPELINE_ROOT
```

Chiều phụ thuộc cụ thể:

- **PixelRL-PARSeq phụ thuộc repo calibrated** để lấy implementation `strhub/PARSeq`, checkpoint PARSeq đã fine-tune, dữ liệu crop biển số và manifest `train/val/test`. Reward OCR-aware và metric CER không thể tái lập đúng nếu thiếu OCR checkpoint tương ứng.
- **Repo calibrated phụ thuộc PixelRL-PARSeq** khi chạy demo đầy đủ. `demo/rl_runtime.py` dùng `RL_PIPELINE_ROOT` để import PixelRL và tìm các checkpoint của selector, bandit, PPO và PixelRL/A2C.
- Nếu chỉ fine-tune hoặc chạy PARSeq cơ bản, repo calibrated có thể chạy độc lập.
- Nếu chỉ huấn luyện phép phục hồi theo pixel, PixelRL có thể chạy với một OCR checkpoint khác. Tuy nhiên kết quả đó không còn là cấu hình của demo này.

Vì vậy, để huấn luyện, đánh giá và chạy toàn bộ demo, hãy clone **cả hai repo cạnh nhau**.

## 1. Cấu trúc thư mục được khuyến nghị

Ví dụ trên Windows:

```text
D:\work\anpr-suite\
├── calibrated-candidate-selector-for-PARSeq\
└── PixelRL-PARSeq\
```

Clone hai repo:

```powershell
$SUITE_ROOT = "D:\work\anpr-suite"
New-Item -ItemType Directory -Force -Path $SUITE_ROOT
Set-Location $SUITE_ROOT

git clone https://github.com/x23d8/calibrated-candidate-selector-for-PARSeq.git
git clone https://github.com/x23d8/PixelRL-PARSeq.git

$CALIBRATED_ROOT = Join-Path $SUITE_ROOT "calibrated-candidate-selector-for-PARSeq"
$PIXELRL_ROOT = Join-Path $SUITE_ROOT "PixelRL-PARSeq"
```

Các biến PowerShell trên chỉ tồn tại trong terminal hiện tại. Khi mở terminal mới, hãy khai báo lại `CALIBRATED_ROOT` và `PIXELRL_ROOT`.

## 2. Yêu cầu hệ thống

- Windows 10/11 hoặc Linux.
- Python 3.10 hoặc 3.11.
- Git.
- GPU NVIDIA và CUDA được khuyến nghị cho training.
- CPU đủ để chạy demo, nhưng 65-view và PixelRL sẽ chậm hơn đáng kể.
- Docker là tùy chọn.

Kiểm tra môi trường:

```powershell
python --version
git --version
nvidia-smi
```

## 3. Cài đặt môi trường Python

Demo tích hợp import trực tiếp mã nguồn từ cả hai repo, vì vậy cách đơn giản nhất là dùng **một virtual environment chung**:

```powershell
Set-Location $SUITE_ROOT
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
```

Cài PyTorch phù hợp với CPU/CUDA của máy theo hướng dẫn chính thức của PyTorch, sau đó cài dependency của project:

```powershell
python -m pip install -r "$CALIBRATED_ROOT\parseq\requirements\train.txt"
python -m pip install -r "$CALIBRATED_ROOT\requirements.txt"
python -m pip install -r "$CALIBRATED_ROOT\demo\requirements.txt"
python -m pip install --no-deps -e "$CALIBRATED_ROOT\parseq"
```

Repo PixelRL dùng chung stack PyTorch, OpenCV, NumPy, PARSeq và các package đã cài ở trên.

Kiểm tra nhanh:

```powershell
python -c "import torch, cv2, joblib; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
python -c "import strhub; print('PARSeq/strhub import OK')"
```

### Chia sẻ PARSeq source cho PixelRL

Một số entrypoint PixelRL tìm `parseq` bên trong chính repo PixelRL. Trên Windows có thể tạo junction để hai repo dùng đúng một bản source:

```powershell
if (-not (Test-Path "$PIXELRL_ROOT\parseq")) {
    New-Item -ItemType Junction `
        -Path "$PIXELRL_ROOT\parseq" `
        -Target "$CALIBRATED_ROOT\parseq"
}

if (-not (Test-Path "$PIXELRL_ROOT\parseq_rl_deblur_data\parseq")) {
    New-Item -ItemType Junction `
        -Path "$PIXELRL_ROOT\parseq_rl_deblur_data\parseq" `
        -Target "$CALIBRATED_ROOT\parseq"
}
```

Không tạo junction nếu đích đã tồn tại. Trên Linux, dùng symbolic link tương đương.

## 4. Chuẩn bị dữ liệu và checkpoint

### 4.1 Dữ liệu fine-tune PARSeq

Script `train_no_refinement/parseq_official_anpr_pipeline.py` nhận:

```text
dataset/
├── images/
│   ├── image_0001.jpg
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

Mỗi CSV cần hai cột:

```csv
image_path,label
images/image_0001.jpg,59D105813
```

Tạo các manifest dùng đường dẫn tuyệt đối cho selector/RL:

```powershell
$DATA_ROOT = "$CALIBRATED_ROOT\dataset"
$MANIFEST_ROOT = "$CALIBRATED_ROOT\outputs\manifests"
New-Item -ItemType Directory -Force -Path $MANIFEST_ROOT

foreach ($split in @("train", "val", "test")) {
    $rows = Import-Csv "$DATA_ROOT\$split.csv" | ForEach-Object {
        [PSCustomObject]@{
            image_path = [IO.Path]::GetFullPath((Join-Path $DATA_ROOT $_.image_path))
            target     = $_.label
            split      = $split
        }
    }
    $rows | Export-Csv "$MANIFEST_ROOT\$split.csv" -NoTypeInformation -Encoding utf8
}
```

Quy tắc quan trọng:

- Một biển số hoặc một ảnh gốc chỉ được thuộc một split.
- Hyperparameter và threshold chỉ được chọn trên `val`.
- Chỉ báo cáo `test` sau khi đã khóa pipeline.
- Chuẩn hóa nhãn phải giống nhau giữa training, selector và demo.

### 4.2 Dữ liệu PixelRL

Dataset builder của PixelRL đọc ba miền màu:

```text
color_filtered/
├── blue/
│   ├── labels.txt
│   └── *.jpg
├── other/
│   ├── labels.txt
│   └── *.jpg
└── yellow/
    ├── labels.txt
    └── *.jpg
```

Mỗi dòng trong `labels.txt` có dạng tab-separated:

```text
image_0001.jpg	59D105813
```

Đặt dữ liệu tại:

```text
PixelRL-PARSeq/parseq_rl_deblur_data/color_filtered/
```

Hoặc chia sẻ dữ liệu từ repo calibrated bằng junction:

```powershell
if (-not (Test-Path "$PIXELRL_ROOT\parseq_rl_deblur_data\color_filtered")) {
    New-Item -ItemType Junction `
        -Path "$PIXELRL_ROOT\parseq_rl_deblur_data\color_filtered" `
        -Target "$CALIBRATED_ROOT\dataset\color_filtered"
}
```

Dữ liệu huấn luyện riêng tư và các checkpoint lớn có thể không được commit vào Git. Nếu một đường dẫn trong hướng dẫn không tồn tại, hãy chuẩn bị artifact đó trước khi chuyển sang bước tiếp theo.

## 5. Fine-tune PARSeq

### 5.1 Chạy bằng script

```powershell
Set-Location $CALIBRATED_ROOT

python train_no_refinement\parseq_official_anpr_pipeline.py `
    --data-root "$CALIBRATED_ROOT\dataset" `
    --output-dir "$CALIBRATED_ROOT\outputs\refinement_finetune" `
    --epochs 30 `
    --batch-size 16 `
    --lr 1e-5 `
    --device cuda
```

Checkpoint chính được dùng trong các bước sau:

```text
outputs/refinement_finetune/best_official_parseq_anpr.pt
```

Có thể chạy notebook tương đương:

```text
refinement_finetune/PARSeq_Official_ANPR_Refinement_Finetune.ipynb
```

Nếu hết VRAM, giảm `--batch-size`. Khi không có GPU, đổi `--device cpu`.

### 5.2 Kiểm tra checkpoint

```powershell
$PARSEQ_CKPT = "$CALIBRATED_ROOT\outputs\refinement_finetune\best_official_parseq_anpr.pt"
Test-Path $PARSEQ_CKPT
```

Lệnh phải trả về `True` trước khi train selector hoặc PixelRL.

## 6. Train 65-view calibrated candidate selector

Selector không học trực tiếp từ ảnh gốc. Quy trình gồm hai pha:

1. **Phase 1:** chạy PARSeq trên 65 view/candidate và ghi prediction, confidence cùng feature của từng view.
2. **Phase 2:** dùng prediction out-of-fold trên validation để calibrate confidence và học quy tắc chọn candidate.

### 6.1 Chuẩn bị manifest

Mỗi manifest cần tối thiểu:

```csv
image_path,target
D:/data/plates/val/001.jpg,59D105813
```

Nếu đã chạy bộ evaluation đầy đủ của repo, có thể dùng trực tiếp:

```text
outputs/refinement_finetune/eval_val_predictions_best_refine.csv
outputs/refinement_finetune/eval_test_predictions_best_refine.csv
```

Hai script Phase 1/2 hiện còn tạo báo cáo riêng cho nhóm ảnh khó lịch sử và cần file:

```text
outputs/testing/irrecoverable_wrong_images_8pipelines/irrecoverable_wrong_images_8pipelines.csv
```

Đây là dependency phục vụ báo cáo, không phải dữ liệu dùng để fit selector. File cần các cột `file,target,best_prediction,image_path,copied_image_path`. Nếu tái lập trên dataset khác, hãy tạo danh sách hard cases tương ứng từ baseline test errors.

### 6.2 Phase 1 — tạo 65-view predictions

Nên chạy entrypoint trong PixelRL repo để artifact nằm đúng cấu trúc mà demo tích hợp sử dụng:

```powershell
Set-Location $PIXELRL_ROOT
$HARD_CASES = "$CALIBRATED_ROOT\outputs\testing\irrecoverable_wrong_images_8pipelines\irrecoverable_wrong_images_8pipelines.csv"
$MANIFEST_ROOT = "$CALIBRATED_ROOT\outputs\manifests"

python reinforcement_learning\run_phase.py phase1 `
    --checkpoint "$PARSEQ_CKPT" `
    --val-manifest "$MANIFEST_ROOT\val.csv" `
    --test-manifest "$MANIFEST_ROOT\test.csv" `
    --irrecoverable-csv "$HARD_CASES" `
    --output-dir "$PIXELRL_ROOT\reinforcement_learning\phase_1_multiscale_tta\results" `
    --batch-size 64 `
    --num-workers 0 `
    --refine-iters 2 `
    --device cuda
```

Nếu prediction của 65 view đã tồn tại và chỉ muốn tạo lại report/selector:

```powershell
python reinforcement_learning\run_phase.py phase1 `
    --checkpoint "$PARSEQ_CKPT" `
    --val-manifest "$MANIFEST_ROOT\val.csv" `
    --test-manifest "$MANIFEST_ROOT\test.csv" `
    --irrecoverable-csv "$HARD_CASES" `
    --output-dir "$PIXELRL_ROOT\reinforcement_learning\phase_1_multiscale_tta\results" `
    --reuse-predictions
```

### 6.3 Phase 2 — fit calibrated selector

```powershell
python reinforcement_learning\run_phase.py phase2 `
    --phase1-dir "$PIXELRL_ROOT\reinforcement_learning\phase_1_multiscale_tta\results" `
    --irrecoverable-csv "$HARD_CASES" `
    --output-dir "$PIXELRL_ROOT\reinforcement_learning\phase_2_calibrated_selector\results" `
    --folds 5
```

Artifact mà web demo tự động tìm:

```text
PixelRL-PARSeq/
└── reinforcement_learning/
    └── phase_2_calibrated_selector/
        └── results/
            └── phase2_selector.joblib
```

Kiểm tra:

```powershell
Test-Path "$PIXELRL_ROOT\reinforcement_learning\phase_2_calibrated_selector\results\phase2_selector.joblib"
```

Không fit calibrator hoặc chọn threshold trên test set. Nếu làm vậy, accuracy test sẽ bị optimistic và không còn là đánh giá holdout.

## 7. Train PixelRL/A2C

PixelRL coi mỗi pixel là một agent. Policy chọn phép biến đổi cục bộ qua nhiều bước; A2C cập nhật policy/value dựa trên reward phục hồi. RMC giúp policy quan sát vùng lân cận thay vì ra quyết định chỉ từ một pixel độc lập.

### 7.1 Tạo dataset

```powershell
Set-Location "$PIXELRL_ROOT\parseq_rl_deblur_data"

python -m rl_deblur.make_dataset `
    --output-dir outputs\rl_deblur\dataset `
    --seed 42 `
    --val-ratio 0.1 `
    --test-ratio 0.1
```

Không thay seed hoặc chia lại dữ liệu giữa các lần thử nếu muốn so sánh công bằng.

### 7.2 Train policy

```powershell
python -m rl_deblur.train `
    --dataset-dir outputs\rl_deblur\dataset `
    --output-dir outputs\rl_deblur `
    --epochs 150 `
    --batch-size 32 `
    --num-steps 5 `
    --gamma 0.95 `
    --lr 1e-4 `
    --channels 64 `
    --rmc-kernel-size 9 `
    --ocr-checkpoint "$PARSEQ_CKPT" `
    --device cuda
```

Checkpoint chính:

```text
parseq_rl_deblur_data/outputs/rl_deblur/checkpoints/best_deblur_agent.pt
```

Tiếp tục một run bị dừng:

```powershell
python -m rl_deblur.train `
    --dataset-dir outputs\rl_deblur\dataset `
    --output-dir outputs\rl_deblur `
    --epochs 150 `
    --ocr-checkpoint "$PARSEQ_CKPT" `
    --resume-checkpoint outputs\rl_deblur\checkpoints\last_training_state.pt `
    --device cuda
```

Mặc định:

```text
--cer-reward-weight 0
--logconf-reward-weight 0
```

được dùng để tái lập objective phục hồi ảnh thuần túy. Muốn huấn luyện OCR-aware, đặt hai trọng số khác 0:

```powershell
python -m rl_deblur.train `
    --dataset-dir outputs\rl_deblur\dataset `
    --output-dir outputs\rl_deblur_ocr_aware `
    --epochs 150 `
    --ocr-checkpoint "$PARSEQ_CKPT" `
    --cer-reward-weight 0.1 `
    --logconf-reward-weight 0.01 `
    --device cuda
```

Hai giá trị trên chỉ là điểm khởi đầu để thử nghiệm, không phải giá trị tối ưu mặc định. Hãy chọn chúng bằng validation CER/sequence accuracy; reward OCR quá lớn có thể làm ảnh trông xấu hơn hoặc khai thác sai confidence của OCR.

### 7.3 Đánh giá PixelRL

Đánh giá validation trước:

```powershell
$PIXELRL_CKPT = "$PIXELRL_ROOT\parseq_rl_deblur_data\outputs\rl_deblur\checkpoints\best_deblur_agent.pt"

python -m rl_deblur.evaluate `
    --dataset-dir outputs\rl_deblur\dataset `
    --agent-checkpoint "$PIXELRL_CKPT" `
    --ocr-checkpoint "$PARSEQ_CKPT" `
    --output-dir outputs\rl_deblur\evaluation_val `
    --split val `
    --device cuda
```

Sau khi đã khóa checkpoint và hyperparameter, đánh giá test:

```powershell
python -m rl_deblur.evaluate `
    --dataset-dir outputs\rl_deblur\dataset `
    --agent-checkpoint "$PIXELRL_CKPT" `
    --ocr-checkpoint "$PARSEQ_CKPT" `
    --output-dir outputs\rl_deblur\evaluation_test `
    --split test `
    --device cuda
```

Nên theo dõi đồng thời:

- Sequence accuracy và CER của PARSeq.
- PSNR/SSIM của ảnh phục hồi.
- Confidence đã calibrate, không chỉ raw confidence.
- Latency và số bước PixelRL.

Ảnh có PSNR tốt hơn chưa chắc cho OCR tốt hơn; mục tiêu cuối của pipeline là đọc đúng chuỗi biển số.

## 8. Optional: train Bandit và PPO restoration policy

Web demo còn hỗ trợ bandit router và PPO. Các policy này chọn pipeline/candidate ở mức toàn ảnh, khác với PixelRL ra hành động theo pixel.

Tạo trajectory cache:

```powershell
Set-Location $PIXELRL_ROOT

python rl_restoration\build_trajectory_cache.py `
    --checkpoint "$PARSEQ_CKPT" `
    --manifest "$MANIFEST_ROOT\train.csv" `
    --split train `
    --output-dir outputs\reproduction\trajectory_cache

python rl_restoration\build_trajectory_cache.py `
    --checkpoint "$PARSEQ_CKPT" `
    --manifest "$MANIFEST_ROOT\val.csv" `
    --split val `
    --output-dir outputs\reproduction\trajectory_cache
```

Train bandit router:

```powershell
python rl_restoration\train_router.py `
    --cache-dir outputs\reproduction\trajectory_cache `
    --seed 123 `
    --output-dir outputs\rl_restoration\router_seed_123
```

Train PPO với bandit làm teacher prior:

```powershell
python rl_restoration\train_ppo.py `
    --cache-dir outputs\reproduction\trajectory_cache `
    --teacher-router outputs\rl_restoration\router_seed_123\best_reward_router.pt `
    --seed 123 `
    --output-dir outputs\rl_restoration\ppo_prior_seed_123
```

Đây là hai đường dẫn mặc định mà demo tìm:

```text
outputs/rl_restoration/router_seed_123/best_reward_router.pt
outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt
```

Liệt kê toàn bộ phase/entrypoint:

```powershell
python reinforcement_learning\run_phase.py --list
```

## 9. Chạy web demo tích hợp

### 9.1 Cấu hình

```powershell
Set-Location $CALIBRATED_ROOT

$env:RL_PIPELINE_ROOT = $PIXELRL_ROOT
$env:PARSEQ_CHECKPOINT = $PARSEQ_CKPT
$env:RL_DEBLUR_CHECKPOINT = $PIXELRL_CKPT
$env:PLATE_DETECTOR_CHECKPOINT = "$CALIBRATED_ROOT\weights\plate_detector.pt"
$env:PARSEQ_DEVICE = "cuda"
$env:PARSEQ_REFINE_ITERS = "2"
$env:PLATE_DETECTOR_CONFIDENCE = "0.25"
```

`RL_DEBLUR_CHECKPOINT` được khai báo tường minh vì checkpoint sau khi train nằm trong subproject `parseq_rl_deblur_data`, trong khi runtime cũng hỗ trợ một layout artifact cũ ở `PixelRL-PARSeq/outputs/rl_deblur`.

### 9.2 Khởi động

```powershell
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Kiểm tra health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Chạy unit/smoke test:

```powershell
python -m unittest demo.test_demo
```

### 9.3 Cách sử dụng

- Upload **ảnh cảnh đầy đủ**: bật tự động phát hiện biển số.
- Upload **ảnh crop biển số**: có thể tắt detector để tránh crop lần hai.
- Chọn `Calibrated Candidate` để chạy selector 65-view.
- Chọn `PixelRL/A2C` để phục hồi ảnh bằng policy theo pixel.
- Chọn bandit/PPO khi các checkpoint tương ứng đã tồn tại.
- So sánh prediction, confidence, ảnh đã xử lý và thời gian chạy.

Một method không xuất hiện hoặc báo unavailable thường có nghĩa artifact của method đó chưa tồn tại dưới `RL_PIPELINE_ROOT`.

## 10. Chạy bằng Docker

Build image từ repo calibrated:

```powershell
docker build -t parseq-anpr-demo "$CALIBRATED_ROOT"
```

Mount PixelRL repo vào container:

```powershell
docker run --rm -p 7860:7860 `
    -e RL_PIPELINE_ROOT=/opt/pixelrl `
    -e RL_DEBLUR_CHECKPOINT=/opt/pixelrl/parseq_rl_deblur_data/outputs/rl_deblur/checkpoints/best_deblur_agent.pt `
    -v "${PIXELRL_ROOT}:/opt/pixelrl:ro" `
    parseq-anpr-demo
```

Mở:

```text
http://127.0.0.1:7860
```

`docker compose up --build` cũng được hỗ trợ, nhưng file Compose hiện dùng volume layout lịch sử `../../rl_pipeline`. Với layout clone được khuyến nghị trong README này, lệnh `docker run` ở trên rõ ràng hơn vì mount đúng `$PIXELRL_ROOT`.

Docker image mặc định hướng tới CPU runtime. Để dùng GPU, máy host cần NVIDIA Container Toolkit và cấu hình PyTorch/CUDA image phù hợp.

## 11. Checklist artifact trước khi chạy demo

```powershell
Test-Path "$CALIBRATED_ROOT\outputs\refinement_finetune\best_official_parseq_anpr.pt"
Test-Path "$CALIBRATED_ROOT\weights\plate_detector.pt"
Test-Path "$PIXELRL_ROOT\reinforcement_learning\phase_2_calibrated_selector\results\phase2_selector.joblib"
Test-Path "$PIXELRL_ROOT\parseq_rl_deblur_data\outputs\rl_deblur\checkpoints\best_deblur_agent.pt"
Test-Path "$PIXELRL_ROOT\outputs\rl_restoration\router_seed_123\best_reward_router.pt"
Test-Path "$PIXELRL_ROOT\outputs\rl_restoration\ppo_prior_seed_123\best_ppo_restoration_policy.pt"
```

Hai artifact cuối là tùy chọn nếu không dùng bandit/PPO. Detector cũng có thể bỏ qua khi input luôn là crop biển số.

## 12. Troubleshooting

### `ModuleNotFoundError: strhub`

```powershell
python -m pip install --no-deps -e "$CALIBRATED_ROOT\parseq"
```

Sau đó kiểm tra junction `PixelRL-PARSeq/parseq` và `PixelRL-PARSeq/parseq_rl_deblur_data/parseq`.

### Demo không thấy PixelRL hoặc selector

Kiểm tra:

```powershell
$env:RL_PIPELINE_ROOT
Test-Path $env:RL_PIPELINE_ROOT
Test-Path $env:RL_DEBLUR_CHECKPOINT
```

Phải đặt biến môi trường trong cùng terminal dùng để chạy Uvicorn.

### `phase2_selector.joblib` không tồn tại

Phase 2 chỉ chạy sau khi Phase 1 đã sinh đủ prediction CSV. Kiểm tra:

```text
PixelRL-PARSeq/reinforcement_learning/phase_1_multiscale_tta/results/
```

Sau đó chạy lại Phase 2 với đúng `--phase1-dir`.

### CUDA out of memory

- Giảm `--batch-size`.
- Giảm số worker.
- Dùng `--device cpu` để kiểm tra chức năng.
- Không chạy nhiều training job trên cùng GPU.

### Prediction tốt trên validation nhưng giảm trên test

Các nguyên nhân phổ biến:

- Fit calibrator/threshold bằng test.
- Trùng ảnh hoặc trùng biển số giữa các split.
- Chọn checkpoint dựa trên test accuracy.
- Domain của ảnh demo khác dữ liệu huấn luyện.
- Raw OCR confidence bị xem như xác suất đã calibrate.

### PixelRL làm ảnh đẹp hơn nhưng OCR kém hơn

PSNR/SSIM và OCR accuracy không hoàn toàn đồng biến. Hãy:

- Chọn checkpoint theo validation OCR metric nếu mục tiêu là ANPR.
- So sánh reward thuần pixel với reward OCR-aware.
- Giới hạn số bước để tránh over-processing.
- Giữ original image như một candidate để selector có thể từ chối ảnh đã phục hồi.

## 13. Entry points chính

```text
calibrated-candidate-selector-for-PARSeq/
├── train_no_refinement/parseq_official_anpr_pipeline.py
├── preprocessing_best_config/benchmark_multiscale_tta.py
├── preprocessing_best_config/benchmark_multiscale_selector_phase2.py
├── demo/app.py
├── demo/rl_runtime.py
└── weights/plate_detector.pt

PixelRL-PARSeq/
├── parseq_rl_deblur_data/rl_deblur/make_dataset.py
├── parseq_rl_deblur_data/rl_deblur/train.py
├── parseq_rl_deblur_data/rl_deblur/evaluate.py
├── reinforcement_learning/run_phase.py
└── rl_restoration/
```

## 14. Tái lập thí nghiệm

Khi báo cáo kết quả, cần lưu:

- Git commit của cả hai repo.
- Checksum checkpoint PARSeq và policy RL.
- Seed chia dữ liệu và seed training.
- Danh sách 65 view/candidate.
- Cấu hình normalization và label canonicalization.
- Hyperparameter selector, PixelRL, bandit/PPO.
- Kết quả validation dùng để chọn model.
- Kết quả test cuối cùng chỉ sau khi khóa cấu hình.

Hai repo phải được version cùng nhau. Thay PARSeq checkpoint, preprocessing candidates hoặc feature schema mà không train lại selector/PixelRL có thể làm calibration và policy không còn hợp lệ.
