# Pipeline nhận diện biển số bằng PARSeq

Pipeline này fine-tune kiến trúc PARSeq chính thức cho bài toán nhận diện ký tự
biển số (ANPR), đánh giá iterative refinement và so sánh các phương pháp tiền xử
lý ảnh trước khi đưa vào mô hình.

Kết luận thực nghiệm hiện tại: `adaptive_noise_3way` và
`clahe_rl_deblur_bilateral` có exact match cao nhất; router adaptive có ít lỗi ký
tự hơn. `clahe_clip1_tile4` vẫn có character accuracy cao nhất và đơn giản hơn.
Các mô hình enhancement tổng quát như Real-ESRGAN, Restormer và Zero-DCE chưa
cải thiện tốt hơn ảnh RGB gốc hoặc CLAHE.

## 1. Kiến trúc hệ thống

```text
Ảnh cắt biển số
      │
      ├── Tiền xử lý tùy chọn: CLAHE, lọc nhiễu, khôi phục ảnh hoặc mô hình ML
      │
      ▼
Resize RGB 32 × 128 và chuẩn hóa
      │
      ▼
ViT encoder: patch 4 × 8 → 128 token ảnh → 12 khối Transformer
      │
      ▼
PARSeq decoder: học thứ tự ký tự bằng Permuted Language Modeling
      │
      ├── Autoregressive decoding
      └── Iterative refinement tùy chọn
      │
      ▼
Chuỗi biển số + độ tin cậy
      │
      ▼
Exact match, character accuracy và CER
```

### Cấu hình PARSeq

| Thành phần | Cấu hình |
| --- | --- |
| Đầu vào | Ảnh RGB, kích thước `32 × 128` |
| Patch embedding | Patch `4 × 8`, tổng cộng 128 patch |
| Kích thước embedding | 384 |
| Encoder | 12 tầng Transformer, 6 đầu chú ý |
| Decoder | 1 tầng Transformer, 12 đầu chú ý |
| Loss | Permuted Language Modeling (PLM) |
| Giải mã | Autoregressive; hỗ trợ `refine_iters = 0, 1, 2, 3` |
| Bảng ký tự ANPR | `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` |
| Độ dài nhãn tối đa | 12 ký tự |

Mã nguồn PARSeq chính thức được đặt trong `parseq/`. Khi `pretrained=True`, mô
hình ban đầu được tải từ bản phát hành chính thức của PARSeq và sau đó fine-tune
bằng PLM loss trên dữ liệu biển số.

Pipeline có bốn luồng chạy độc lập:

1. Fine-tune bằng CLI, mặc định không dùng iterative refinement.
2. Fine-tune và quét iterative refinement bằng notebook.
3. Benchmark 99 cấu hình tiền xử lý, tổ hợp và routing thích ứng.
4. Benchmark các mô hình enhancement dùng weight chính chủ.

## 2. Cấu trúc thư mục

```text
parseq_official_pipeline/
├── dataset/                         # Dữ liệu dùng bởi notebook refinement
├── outputs/                         # Checkpoint, log và kết quả đánh giá
├── parseq/                          # Mã nguồn PARSeq chính thức
├── preprocessing_best_config/
│   ├── preprocessing.py             # Các cấu hình tiền xử lý ảnh
│   ├── find_best_preprocessing_config.py
│   ├── ml_official_preprocessing_benchmark.py
│   ├── EXPERIMENT_REPORT.md
│   ├── COMBINATION_EXPERIMENT_REPORT.md
│   ├── ADAPTIVE_PREPROCESSING_REPORT.md
│   └── ML_OFFICIAL_BENCHMARK_REPORT.md
├── refinement_finetune/
│   └── PARSeq_Official_ANPR_Refinement_Finetune.ipynb
├── train_no_refinement/
│   └── parseq_official_anpr_pipeline.py
├── requirements.txt
└── README.md
```

Tất cả lệnh trong tài liệu này được chạy tại thư mục
`parseq_official_pipeline/`.

## 3. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.10 trở lên và GPU NVIDIA. Có thể chạy bằng CPU,
nhưng quá trình fine-tune và benchmark mô hình ML sẽ chậm hơn đáng kể.

### 3.1. Tạo môi trường ảo trên Windows

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Để chạy notebook:

```powershell
pip install jupyterlab ipykernel
python -m ipykernel install --user --name parseq-anpr --display-name "PARSeq ANPR"
```

Các dependency bổ sung cho benchmark ML:

```powershell
pip install basicsr==1.4.2 addict einops gdown
```

Kiểm tra PyTorch và GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `CUDA: False` dù máy có GPU NVIDIA, hãy cài lại bản PyTorch tương thích với
CUDA của máy trước khi chạy fine-tune.

## 4. Chuẩn bị dữ liệu

### 4.1. Dữ liệu cho script CLI

Thư mục dữ liệu phải có ba tệp `train.csv`, `val.csv` và `test.csv`:

```text
du_lieu_bien_so/
├── images/
│   ├── plate_0001.png
│   └── plate_0002.png
├── train.csv
├── val.csv
└── test.csv
```

Mỗi CSV cần hai cột:

```csv
image_path,label
images/plate_0001.png,51A4032
images/plate_0002.png,77H61141
```

- `image_path` là đường dẫn tương đối tính từ thư mục dữ liệu.
- `label` được chuyển thành chữ hoa và chỉ giữ ký tự `0-9`, `A-Z`.
- Nhãn rỗng hoặc dài hơn 12 ký tự sẽ bị loại.
- Phải khóa tập test từ đầu; không dùng test để chọn checkpoint hoặc tham số tiền
  xử lý.

### 4.2. Dữ liệu cho notebook refinement

Notebook hiện ghép ba nguồn:

```text
dataset/
├── color_filtered/                  # Biển số thông thường
└── update_label/
    ├── labled_quandoi/              # Biển số quân đội
    └── label_ngoaigiao/             # Biển số ngoại giao
```

Cấu hình nguồn dữ liệu nằm trong biến `dataset_sources` ở cell cấu hình của
notebook. Notebook hỗ trợ nguồn dạng `labels.txt` theo thư mục và nguồn CSV đã
được thu thập, rà soát nhãn. Nếu thay đổi vị trí dữ liệu, chỉ sửa
`DATASET_DIR` và `dataset_sources`; không sửa đường dẫn ở nhiều cell khác nhau.

## 5. Fine-tune PARSeq bằng CLI

Luồng này dùng mã triển khai trong `parseq/strhub`, tải trọng số tiền huấn luyện
chính thức và mặc định đặt `refine_iters=0`.

```powershell
$DATA_ROOT = "D:\du_lieu\du_lieu_bien_so"

python train_no_refinement\parseq_official_anpr_pipeline.py `
  --data-root $DATA_ROOT `
  --output-dir outputs\train_no_refinement `
  --epochs 30 `
  --batch-size 16 `
  --lr 1e-5 `
  --refine-iters 0 `
  --device cuda
```

Để huấn luyện và đánh giá nhất quán với CLAHE tốt nhất hiện tại:

```powershell
python train_no_refinement\parseq_official_anpr_pipeline.py `
  --data-root $DATA_ROOT `
  --output-dir outputs\train_clahe_clip1_tile4 `
  --epochs 30 `
  --batch-size 16 `
  --refine-iters 0 `
  --preprocess `
  --preprocess-config clahe_clip1_tile4 `
  --device cuda
```

Các tham số thường dùng:

| Tham số | Mặc định | Ý nghĩa |
| --- | ---: | --- |
| `--epochs` | 5 | Số epoch fine-tune |
| `--batch-size` | 16 | Batch size train, validation và test |
| `--lr` | `1e-5` | Learning rate của AdamW |
| `--weight-decay` | `1e-4` | Weight decay của AdamW |
| `--refine-iters` | 0 | Số vòng iterative refinement |
| `--preprocess` | tắt | Bật tiền xử lý ảnh |
| `--preprocess-config` | `train_baseline` | Tên cấu hình trong `preprocessing.py` |
| `--no-augment` | tắt | Tắt augmentation khi train |
| `--device` | tự động | Ví dụ: `cuda`, `cuda:0` hoặc `cpu` |

Xem toàn bộ tham số:

```powershell
python train_no_refinement\parseq_official_anpr_pipeline.py --help
```

Kết quả được ghi vào `--output-dir`:

- `best_official_parseq_anpr.pt`: checkpoint có exact match validation cao nhất.
- `history.csv`: loss và metric theo epoch.
- `test_predictions.csv`: dự đoán trên từng ảnh test.
- `summary.json`: cấu hình, epoch tốt nhất và metric cuối cùng.

## 6. Fine-tune và đánh giá refinement bằng notebook

Mở notebook:

```powershell
jupyter lab refinement_finetune\PARSeq_Official_ANPR_Refinement_Finetune.ipynb
```

Sau khi chọn kernel `PARSeq ANPR`:

1. Kiểm tra `DATASET_DIR` và `dataset_sources` trong cell cấu hình.
2. Điều chỉnh `epochs`, `batch_size`, `preprocess` và `refine_iters` nếu cần.
3. Chạy `Run All` theo đúng thứ tự cell.
4. Không chạy riêng cell test để thử nhiều cấu hình trên tập test.

Notebook thực hiện:

- tải trọng số tiền huấn luyện PARSeq chính thức;
- ghép và chuẩn hóa nhiều loại biển số;
- đánh giá `refine_iters = 0, 1, 2, 3` trước fine-tune;
- fine-tune bằng PLM loss;
- chọn checkpoint theo exact match validation;
- chọn số vòng refinement trên validation;
- báo cáo các cấu hình refinement đã định trước trên test, nhưng chỉ dùng kết quả
  validation để chọn cấu hình tốt nhất;
- xuất các trường hợp nhận diện sai.

Mỗi lần chạy tạo thư mục:

```text
outputs/refinement_finetune_YYYYMMDD_HHMMSS/
```

Các tệp kết quả quan trọng:

- `best_official_parseq_anpr.pt`: checkpoint tốt nhất.
- `refinement_sweep_val.csv`: kết quả refinement trên validation.
- `refinement_sweep_test.csv`: xác nhận cuối trên test.
- `eval_val_predictions_best_refine.csv`: manifest và dự đoán validation.
- `eval_test_predictions_best_refine.csv`: manifest và dự đoán test.
- `eval_*_by_plate_type.csv`: metric theo loại biển số.
- `wrong_eval_images/`: ảnh nhận diện sai để phân tích lỗi.
- `summary.json`: cấu hình và toàn bộ kết quả chính.

## 7. Benchmark tiền xử lý ảnh truyền thống

Benchmark sử dụng checkpoint và đúng manifest validation/test của một lần chạy
notebook. Tất cả cấu hình được xếp hạng trên validation; chỉ các finalist mới
được xác nhận trên test.

```powershell
$RUN_DIR = "outputs\refinement_finetune_YYYYMMDD_HHMMSS"

python preprocessing_best_config\find_best_preprocessing_config.py `
  --run-dir $RUN_DIR `
  --refine-iters 2 `
  --top-k 3 `
  --batch-size 64 `
  --device cuda
```

Nếu bỏ `--run-dir`, script tự tìm thư mục `refinement_finetune*` mới nhất có đủ
checkpoint và manifest validation.

Kết quả mặc định nằm trong `outputs/preprocessing_course_benchmark/`:

- `validation_results.csv`: xếp hạng toàn bộ cấu hình.
- `best_preprocessing_config.json`: cấu hình tốt nhất trên validation.
- `test_finalists_results.csv`: kết quả finalist trên test khóa.
- `predictions_<split>_<config>.csv`: dự đoán từng ảnh.
- `REPORT.md` và `summary.json`: báo cáo và thông tin tái lập.

Báo cáo phân tích đầy đủ: [EXPERIMENT_REPORT.md](preprocessing_best_config/EXPERIMENT_REPORT.md).

Các tổ hợp deblur, low-pass, tăng cường biên và cô lập ký tự được phân tích tại
[COMBINATION_EXPERIMENT_REPORT.md](preprocessing_best_config/COMBINATION_EXPERIMENT_REPORT.md).

Phân tích đặc tính dataset và routing thích ứng:
[ADAPTIVE_PREPROCESSING_REPORT.md](preprocessing_best_config/ADAPTIVE_PREPROCESSING_REPORT.md).

## 8. Benchmark tiền xử lý bằng mô hình ML

### 8.1. Chuẩn bị mã nguồn và trọng số chính chủ

Script hiện hỗ trợ trọng số chính chủ của:

| Mô hình | Nhiệm vụ | Vị trí trọng số |
| --- | --- | --- |
| Real-ESRGAN x2plus | Super-resolution | `.cache/official_ml/weights/Real-ESRGAN/RealESRGAN_x2plus.pth` |
| Restormer | Motion deblurring | `.cache/official_ml/weights/Restormer/motion_deblurring.pth` |
| Zero-DCE | Tăng sáng ảnh thiếu sáng | `.cache/official_ml/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth` |

Chạy một lần các lệnh sau từ thư mục gốc dự án:

```powershell
New-Item -ItemType Directory -Force .cache\official_ml\weights\Real-ESRGAN
New-Item -ItemType Directory -Force .cache\official_ml\weights\Restormer

git clone https://github.com/swz30/Restormer.git .cache\official_ml\Restormer
git -C .cache\official_ml\Restormer checkout 68dc6ac472db26f16361150cb7a96a1bc87da93f

git clone https://github.com/Li-Chongyi/Zero-DCE.git .cache\official_ml\Zero-DCE
git -C .cache\official_ml\Zero-DCE checkout e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd

Invoke-WebRequest `
  -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" `
  -OutFile ".cache\official_ml\weights\Real-ESRGAN\RealESRGAN_x2plus.pth"

gdown 1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L `
  -O ".cache\official_ml\weights\Restormer\motion_deblurring.pth"
```

Các nguồn chính thức:

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Restormer](https://github.com/swz30/Restormer)
- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)

Không tải trọng số từ nguồn sao chép không chính thức. Script kiểm tra SHA-256
trước khi nạp và dừng ngay nếu file không đúng với trọng số đã xác minh. Nguồn
tải, commit và hash đầy đủ được ghi trong
`outputs/ml_official_preprocessing_benchmark/official_model_provenance.json` sau
khi chạy.

### 8.2. Chạy benchmark

Chạy benchmark:

```powershell
$RUN_DIR = "outputs\refinement_finetune_YYYYMMDD_HHMMSS"

python preprocessing_best_config\ml_official_preprocessing_benchmark.py `
  --run-dir $RUN_DIR `
  --refine-iters 2 `
  --top-k-ml 2 `
  --batch-size 64 `
  --ml-batch-size 8 `
  --device cuda
```

Kết quả mặc định nằm trong `outputs/ml_official_preprocessing_benchmark/`.
Báo cáo chi tiết: [ML_OFFICIAL_BENCHMARK_REPORT.md](preprocessing_best_config/ML_OFFICIAL_BENCHMARK_REPORT.md).

## 9. Kết quả tham chiếu hiện tại

Checkpoint epoch 26, `refine_iters=2`, validation 397 ảnh và test khóa 411 ảnh:

| Phương pháp | Exact match test | Character accuracy test |
| --- | ---: | ---: |
| `adaptive_noise_3way` | **93,43%** | 98,99% |
| `clahe_rl_deblur_bilateral` | **93,43%** | 98,93% |
| `clahe_clip1_tile4` | **93,19%** | **99,08%** |
| `raw_rgb` | **93,19%** | 98,99% |
| `zero_dce` | 92,94% | 98,96% |
| `restormer_motion_deblur_native` | 92,94% | 98,96% |
| `train_baseline` | 91,97% | 98,87% |

Khuyến nghị hiện tại:

- dùng `adaptive_noise_3way` làm ứng viên nếu ưu tiên exact match;
- dùng `clahe_clip1_tile4` nếu ưu tiên character accuracy và tốc độ;
- giữ `raw_rgb` làm mốc kiểm soát;
- không đưa Real-ESRGAN, Restormer hoặc Zero-DCE vào toàn bộ luồng inference;
- xác nhận lại trên một tập giữ lại độc lập trước khi đưa vào vận hành vì khoảng
  tin cậy bootstrap của mức tăng CLAHE vẫn cắt qua 0.

## 10. Quy tắc đánh giá

- Chọn epoch, `refine_iters` và preprocessing bằng validation.
- Chỉ dùng test để xác nhận cấu hình đã khóa.
- Luôn báo cáo đồng thời exact match, character accuracy và CER.
- So sánh dự đoán theo cặp trên cùng ảnh; ưu tiên bootstrap theo cặp khi dữ liệu
  test nhỏ.
- Không so sánh các phương pháp trên những split hoặc checkpoint khác nhau.

Trong đó:

- **Exact match**: tỷ lệ biển số được nhận diện đúng toàn bộ chuỗi.
- **Character accuracy**: `1 - tổng edit distance / tổng số ký tự`.
- **CER**: `tổng edit distance / tổng số ký tự`.

## 11. Xử lý lỗi thường gặp

### Không tìm thấy `train.csv`, `val.csv` hoặc `test.csv`

Kiểm tra `--data-root`. Script CLI yêu cầu ba CSV nằm trực tiếp trong thư mục
được truyền vào.

### CUDA hết bộ nhớ

Giảm `--batch-size`. Với benchmark ML, giảm thêm `--ml-batch-size`; các biến thể
chạy ở kích thước ảnh gốc luôn dùng batch size 1.

### Không tìm thấy checkpoint khi benchmark

Truyền đúng `--run-dir`, hoặc chỉ định riêng:

```powershell
python preprocessing_best_config\find_best_preprocessing_config.py `
  --checkpoint "duong_dan\best_official_parseq_anpr.pt" `
  --val-manifest "duong_dan\eval_val_predictions_best_refine.csv" `
  --test-manifest "duong_dan\eval_test_predictions_best_refine.csv"
```

Ba file phải thuộc cùng một lần chạy.

### Thiếu trọng số của mô hình ML

Đặt đúng trọng số chính chủ vào các vị trí ở mục 8. Không đổi hash trong mã nguồn
để bỏ qua bước kiểm tra tính toàn vẹn.

### Kết quả chạy lại khác nhau

Giữ nguyên `--seed`, split dữ liệu, checkpoint, preprocessing và
`refine_iters`. Một số CUDA kernel vẫn có thể tạo sai khác số học rất nhỏ.
