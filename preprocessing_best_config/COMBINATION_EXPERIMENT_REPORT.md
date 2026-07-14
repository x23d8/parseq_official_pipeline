# Thử nghiệm tổ hợp tiền xử lý ảnh cho PARSeq

## Kết luận

Đã bổ sung và đánh giá 26 tổ hợp mới thuộc ba nhóm:

1. Khôi phục ảnh: Wiener hoặc Richardson–Lucy deblur, sau đó lọc Gaussian hay
   bilateral để giảm ringing và nhiễu khuếch đại.
2. Tăng cường nét: homomorphic, unsharp, Sobel fusion và top-hat/black-hat hai
   cực sáng tối.
3. Cô lập ký tự: crop vùng chứa connected components, tạo mặt nạ ký tự hoặc làm
   mờ nền ngoài ký tự.

Tổ hợp có exact match test cao nhất là:

```text
grayscale → CLAHE 1.0/tile 4×4 → Richardson–Lucy 3 vòng
→ Gaussian PSF 3×3, sigma 0.7 → bilateral 3×3, sigma 25
```

Tên cấu hình: `clahe_rl_deblur_bilateral`.

## Giao thức

- Checkpoint: epoch 26, `refine_iters=2`.
- Validation: 397 ảnh; test khóa: 411 ảnh.
- Sweep: 26 tổ hợp mới và 4 mốc mạnh từ benchmark trước.
- Xếp hạng bằng exact match validation trước, character accuracy sau.
- Chỉ 8 cấu hình đứng đầu validation và baseline được xác nhận trên test.
- Bootstrap theo cặp dùng cùng ảnh giữa hai phương pháp.

## Kết quả chính

| Phương pháp | Val exact | Val char acc | Test exact | Test char acc |
| --- | ---: | ---: | ---: | ---: |
| `clahe_rl_deblur_bilateral` | 93,20% | 98,87% | **93,43%** | 98,93% |
| `clahe_clip1_tile4` | 93,20% | 98,74% | 93,19% | **99,08%** |
| `rl_deblur_bilateral_lowpass` | **93,95%** | 98,74% | 92,21% | 98,90% |
| `homomorphic_filter` | 93,70% | **99,05%** | 92,46% | 98,87% |
| `train_baseline` | 92,70% | 98,65% | 91,97% | 98,87% |

So với baseline, `clahe_rl_deblur_bilateral`:

- tăng exact match 1,4599 điểm phần trăm;
- tăng character accuracy 0,0596 điểm phần trăm;
- sửa đúng 9 ảnh và làm sai 3 ảnh baseline vốn nhận diện đúng;
- khoảng tin cậy bootstrap 95% exact xấp xỉ [0,00; +3,16] điểm phần trăm;
- khoảng tin cậy character accuracy vẫn cắt qua 0.

So với `clahe_clip1_tile4`, tổ hợp mới:

- tăng ròng 1/411 ảnh đúng, tương đương +0,2433 điểm phần trăm exact match;
- sửa đúng 2 ảnh nhưng làm sai 1 ảnh CLAHE vốn đúng;
- giảm 0,1489 điểm phần trăm character accuracy do một số lỗi sai nhiều ký tự;
- CI 95% exact [-0,49; +1,22] và character accuracy [-0,42; +0,06] điểm phần
  trăm, nên chưa có bằng chứng đủ mạnh rằng tổ hợp mới tốt hơn CLAHE.

## Đánh giá từng nhóm

### Deblur rồi low-pass

Richardson–Lucy kết hợp bilateral là nhánh tốt nhất. Tuy nhiên cấu hình không có
CLAHE thắng validation nhưng không giữ mức tăng trên test. Điều này cho thấy PSF
giả định chỉ phù hợp với một phần ảnh; không nên áp deblur mạnh cho mọi crop.

Wiener deconvolution thấp hơn baseline ở phần lớn cấu hình. Inverse filter tạo
ringing khi kernel giả định không khớp chuyển động hoặc defocus thật.

### Tách biên và tăng nét ký tự

Homomorphic kết hợp unsharp hoặc dual-stroke cho validation tốt, nhưng test không
vượt CLAHE. Sobel fusion không tăng test exact và làm character accuracy giảm
nhẹ. PARSeq đã học đặc trưng nét từ ảnh tự nhiên nên biên nhân tạo không cung cấp
thêm thông tin ổn định.

### Tách vùng và mặt nạ ký tự

| Cấu hình | Val exact | Val char acc |
| --- | ---: | ---: |
| `component_fusion_gray` | 87,66% | 97,00% |
| `content_crop_homomorphic` | 85,89% | 96,57% |
| `content_crop_clahe` | 85,64% | 96,23% |
| `content_crop_gray` | 85,39% | 96,02% |
| `component_mask_gray` | 70,53% | 90,54% |

Tách ký tự bằng connected components không phù hợp làm preprocessing toàn cục:

- nét ký tự có thể bị chia thành nhiều component hoặc dính với khung biển;
- biển hai dòng làm thứ tự và chiều cao component thay đổi;
- crop lại làm thay đổi bố cục mà checkpoint đã học;
- mặt nạ làm mất màu, anti-alias và ngữ cảnh nền hữu ích.

Muốn OCR từng ký tự riêng, cần một pipeline khác gồm detector/segmenter ký tự,
sắp xếp ký tự theo một hoặc hai dòng và một classifier ký tự được huấn luyện trên
crop đơn ký tự. Checkpoint PARSeq hiện tại là bộ nhận diện toàn chuỗi, không phải
classifier ký tự rời.

## Khuyến nghị

- Nếu ưu tiên exact match tuyệt đối, giữ `clahe_rl_deblur_bilateral` làm ứng viên
  thử nghiệm trên một holdout mới.
- Nếu ưu tiên character accuracy, tốc độ và độ đơn giản, tiếp tục dùng
  `clahe_clip1_tile4`.
- Chưa thay cấu hình production vì mức hơn CLAHE chỉ là một ảnh và CI cắt qua 0.
- Hướng tiếp theo hợp lý là quality router: chỉ gọi deblur khi ảnh thực sự mờ,
  thay vì áp Richardson–Lucy cho toàn bộ ảnh.

Kết quả và prediction từng ảnh nằm trong
`outputs/testing/preprocessing_combinations_benchmark/`.

