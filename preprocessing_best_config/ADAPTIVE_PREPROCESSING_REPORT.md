# Tiền xử lý thích ứng theo đặc tính dataset

## Mục tiêu

Một pipeline cố định không phù hợp cho toàn bộ dataset: homomorphic tốt trên ảnh
tối hoặc tương phản thấp, CLAHE tốt trên ảnh sáng trung bình, còn
Richardson–Lucy kết hợp bilateral tốt hơn ở nhiều crop sáng/nét. Thử nghiệm này
dùng thống kê ảnh không cần nhãn để chọn pipeline riêng cho từng crop.

## Đặc tính validation

Validation gồm 397 ảnh với phân bố không đồng nhất:

| Thuộc tính | P10 | Trung vị | P90 |
| --- | ---: | ---: | ---: |
| Chiều rộng | 59 px | 81 px | 162 px |
| Chiều cao | 25 px | 58 px | 86 px |
| Tỷ lệ rộng/cao | 1,11 | 1,43 | 3,55 |
| Độ sáng trung bình | 92,94 | 145,24 | 174,15 |
| Độ tương phản | 48,92 | 65,03 | 86,69 |
| Laplacian variance | 1.654 | 4.092 | 18.683 |
| Saturation trung bình | 19,45 | 43,17 | 82,11 |

Khoảng 40% ảnh có tỷ lệ dưới 1,25, chủ yếu là crop cao hoặc biển hai dòng; 32%
ảnh có tỷ lệ trên 2,2, chủ yếu là biển một dòng. Một phần đáng kể crop rất nhỏ:
10% ảnh thấp không quá 25 px.

Phân tích subgroup trên validation cho thấy:

- ảnh tối: homomorphic đạt 93,98% exact, baseline 91,57%;
- ảnh sáng: RL + bilateral đạt 92,31%, baseline 90,21%;
- crop rộng: RL đạt 96,12%, baseline 94,57%;
- crop vuông: RL đạt 90,74%, baseline 87,96%;
- biển quân đội: RL/homomorphic đạt 92,31%, baseline 84,62%;
- ảnh có saturation cao: homomorphic đạt 89,71%, baseline 86,76%.

## Cấu hình mới

Đã thêm 26 cấu hình, nâng tổng số cấu hình trong thư viện lên 99:

- 8 router thích ứng theo quality, brightness, aspect ratio, kích thước, noise và
  tỷ lệ pixel tối;
- 5 cấu hình upscale có điều kiện cho crop nhỏ;
- 6 mức CLAHE nhẹ với tile 2×2 hoặc 4×4;
- 7 chuỗi bổ sung cho chiếu sáng không đều, kênh green và deblur.

Policy `adaptive_quality_cv` được khóa bằng cross-validation 5-fold. Ngưỡng được
chọn ổn định ở nhiều seed, nhưng vẫn không tổng quát tốt nhất trên test. Policy
`adaptive_noise_3way` đơn giản và ổn định hơn:

```text
noise <= 5      → homomorphic_filter
5 < noise <= 10 → clahe_rl_deblur_bilateral
noise > 10      → rl_deblur_bilateral_lowpass
```

`noise` là trung vị trị tuyệt đối của phần dư giữa ảnh xám và Gaussian blur 3×3.

## Kết quả

| Phương pháp | Val exact | Val char acc | Test exact | Test char acc |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_quality_cv` | **94,46%** | **99,14%** | 92,70% | 98,87% |
| `adaptive_noise_3way` | **94,46%** | 99,05% | **93,43%** | 98,99% |
| `adaptive_brightness_3way` | **94,46%** | 99,02% | 92,94% | 98,96% |
| `clahe_rl_deblur_bilateral` | 93,20% | 98,87% | **93,43%** | 98,93% |
| `clahe_clip1_tile4` | 93,20% | 98,74% | 93,19% | **99,08%** |
| `train_baseline` | 92,70% | 98,65% | 91,97% | 98,87% |

So với baseline, `adaptive_noise_3way`:

- tăng 1,4599 điểm phần trăm exact match;
- tăng 0,1191 điểm phần trăm character accuracy;
- sửa đúng 10 ảnh và làm sai 4 ảnh baseline vốn đúng;
- CI bootstrap 95% của cả hai mức tăng vẫn cắt qua 0.

So với CLAHE đơn:

- tăng ròng 1/411 ảnh đúng, tương đương +0,2433 điểm phần trăm exact;
- sửa đúng 4 ảnh và làm sai 3 ảnh CLAHE vốn đúng;
- giảm 0,0893 điểm phần trăm character accuracy;
- CI 95% exact [-0,97; +1,46] và character accuracy [-0,33; +0,12] điểm phần
  trăm.

Router phân phối ảnh như sau:

| Nhánh | Validation | Test | Test exact trong nhánh |
| --- | ---: | ---: | ---: |
| `homomorphic_filter` | 91 | 102 | 88,24% |
| `clahe_rl_deblur_bilateral` | 191 | 175 | 94,86% |
| `rl_deblur_bilateral_lowpass` | 115 | 134 | 95,52% |

## Nhận xét

- Router theo noise giữ exact của chuỗi tốt nhất nhưng giảm các lỗi nhiều ký tự,
  vì không áp cùng một kiểu deblur cho mọi ảnh.
- Router `quality_cv` thắng validation nhưng giảm trên test, cho thấy ngưỡng nhiều
  biến dễ khớp quá mức với 397 ảnh validation.
- Upscale crop nhỏ giúp validation nhẹ (`small_cubic_clahe` đạt 93,70%) nhưng
  chưa đủ cao để được chọn sang test.
- CLAHE clip 0,5–1,25 và tile 2×2/4×4 gần như hòa nhau; tiếp tục quét tham số nhỏ
  hơn không có nhiều giá trị.

## Khuyến nghị

`adaptive_noise_3way` là ứng viên tốt nhất nếu ưu tiên exact match mà vẫn muốn
giảm lỗi ký tự so với chuỗi deblur cố định. `clahe_clip1_tile4` vẫn phù hợp hơn
nếu ưu tiên character accuracy, tốc độ và cách triển khai đơn giản.

Test 411 ảnh đã được dùng để xác nhận nhiều vòng thử nghiệm, vì vậy không còn là
holdout hoàn toàn chưa quan sát. Không chọn thêm ngưỡng dựa trên test này. Cần một
holdout mới trước khi thay preprocessing trong môi trường vận hành.

Kết quả đầy đủ và routing audit nằm trong
`outputs/testing/preprocessing_adaptive_benchmark/`.

