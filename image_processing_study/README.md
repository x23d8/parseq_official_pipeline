# Nghiên cứu so sánh các phương pháp xử lý ảnh cho ANPR

Đồ án môn **Xử lý ảnh**: ablation study so sánh nhiều kỹ thuật xử lý ảnh (bám khung chương trình
môn học) áp dụng cho bài toán đọc biển số xe, trên **cùng một kiến trúc model**. Khác với
`preprocessing_best_config/` (chỉ đổi cách xử lý ảnh lúc đánh giá PARSeq đã pretrained),
module này **train riêng một model từ đầu cho mỗi phương pháp xử lý ảnh**, để đo đúng câu hỏi
"phương pháp nào giúp model học tốt hơn" thay vì chỉ đo độ bền của một model có sẵn.

## Vì sao không dùng PARSeq

PARSeq (`parseq/`) đã được pretrain rất mạnh nên khác biệt giữa các cách xử lý ảnh đầu vào gần như
bị san phẳng khi fine-tune. Module này xây một **CRNN + CTC** nhẹ (~2M tham số, kiến trúc kinh điển
của Shi et al.), **train from scratch** cho từng phương pháp -- đủ nhạy để chất lượng ảnh đầu vào
thật sự ảnh hưởng tới độ chính xác, đúng tinh thần một thí nghiệm nghiên cứu.

## Hai thí nghiệm

- **Thí nghiệm A** (`run_experiment_a.py`) -- train 1 CRNN riêng cho mỗi phương pháp trong
  `methods.py` (cùng kiến trúc/seed/hyperparameter, cùng 1 split train/val/test cố định từ
  `dataset.build_split`), so sánh exact-match accuracy / CER trên tập test. Đây là bảng kết quả
  chính, trả lời "phương pháp xử lý ảnh nào tốt nhất cho OCR biển số".
- **Thí nghiệm B** (`run_experiment_b.py`) -- vì ảnh biển số thật không có "ảnh sạch" đã biết để so
  sánh, thí nghiệm này tạo suy giảm tổng hợp có kiểm soát (`degrade.py`: Gaussian/motion/defocus
  blur, Gaussian noise) trên đúng tập test của Thí nghiệm A, rồi đo PSNR/SSIM và OCR accuracy (qua
  model `raw` đã train ở Thí nghiệm A) của các phương pháp phục hồi ảnh -- gồm cả agent RL deblur
  (`rl_deblur/`) để so sánh với các phương pháp cổ điển.

## 13 phương pháp xử lý ảnh (`methods.py`)

| Method | Kỹ thuật | Chương môn học |
| --- | --- | --- |
| `raw` | Không xử lý (đối chứng), resize bilinear | baseline |
| `bicubic_resize` | Resize bicubic thay vì bilinear | 7.1 Sampling & Interpolation |
| `hist_eq` | Global histogram equalization | 1.1 Gray-level processing |
| `otsu_binary` | Otsu threshold -> nhị phân | 1.2 Binary image processing |
| `clahe` | Adaptive histogram equalization | 2.1 Linear filtering / enhancement |
| `median_denoise` | Median filter | 2.2 Nonlinear filtering |
| `bilateral_denoise` | Bilateral filter | 2.2 Nonlinear filtering |
| `morph_tophat` | White + black top-hat | 2.3 Morphological filtering |
| `freq_highboost` | High-boost filter qua FFT | 2.5 Frequency-domain filtering |
| `homomorphic` | Homomorphic filtering | 2.5 / 3.1 Restoration model |
| `wavelet_denoise` | Wavelet soft-threshold (BayesShrink) | 3.5 Wavelet denoising |
| `wiener_restore` | Wiener deconvolution (PSF giả định) | 3.1/3.4/3.6 Restoration & MMSE |
| `rl_deblur_restore` | Agent RL (PixelRL + A2C) từ `rl_deblur/` | so sánh deep-learning restoration |

`rl_deblur_restore` chỉ được thêm vào nếu tìm thấy checkpoint đã train
(`outputs/rl_deblur/checkpoints/best_deblur_agent.pt`); nếu không có, method này tự bỏ qua (có log
cảnh báo) thay vì làm hỏng cả sweep.

## Biến kiểm soát quan trọng nhất

Mọi phương pháp dùng **chung đúng 1 bộ train/val/test split** (`dataset.build_split`, seed cố định
`ocr_train.SPLIT_SEED = 42`) từ `color_filtered/{blue,other,yellow}/`. Chỉ cách xử lý pixel thay
đổi giữa các lần train -- đây là điều kiện bắt buộc để bảng so sánh có ý nghĩa khoa học.

## Chạy local (smoke test, CPU)

```bash
python -m image_processing_study.run_experiment_a \
  --methods raw clahe wavelet_denoise --no-include-rl \
  --epochs 2 --batch-size 16 --limit-train 60 --limit-val 20 --limit-test 20 --device cpu

python -m image_processing_study.run_experiment_b \
  --raw-checkpoint outputs/image_processing_study/experiment_a/raw/best_model.pt \
  --no-include-rl --limit 20 --device cpu
```

Dùng để bắt lỗi cú pháp/shape/CTC trước khi chạy thật -- không kỳ vọng độ chính xác cao với dữ liệu
bị giới hạn (`--limit-*`).

## Chạy thật trên Colab

Notebook `IMAGE_PROCESSING_STUDY_Colab.ipynb` chỉ giải nén 1 file zip rồi `import image_processing_study`
trực tiếp -- không dùng `%%writefile` (khác với `rl_deblur/RL_Deblur_Colab.ipynb`), nên zip cần gộp
cả code lẫn data:

```bash
zip -r parseq_ip_study_data.zip \
  image_processing_study color_filtered rl_deblur \
  outputs/rl_deblur/checkpoints/best_deblur_agent.pt \
  -x "*/__pycache__/*" "*.ipynb"
```

Mỗi lần sửa code ở local, zip + upload lại là đủ, không cần sửa gì trong notebook.

Upload file lên Google Drive rồi mở notebook trên Colab (Runtime > GPU T4). Notebook chạy cả 2 thí
nghiệm, sinh ảnh minh họa + biểu đồ so sánh, và nén `outputs/image_processing_study/` gửi lại Drive.

## Output

- `outputs/image_processing_study/experiment_a/<method>/{best_model.pt,history.csv,test_predictions.csv,summary.json}`
- `outputs/image_processing_study/experiment_a/comparison.csv` -- bảng xếp hạng chính
- `outputs/image_processing_study/experiment_b/comparison.csv` -- PSNR/SSIM + OCR theo phương pháp phục hồi
- `outputs/image_processing_study/samples/` -- lưới ảnh before/after + biểu đồ cột
