# Báo cáo loại `normal` khỏi fine-tune PARSeq

## Kết luận

Checkpoint canonical `outputs/refinement_finetune/best_official_parseq_anpr.pt` không dùng nguồn
`vietnam_normal` hoặc loại biển `normal`. Artifact cũ có `normal` đã được thay bằng checkpoint của
run `refinement_finetune_20260710_142307`.

**Trạng thái hiện tại:** checkpoint không bị leakage do ảnh trùng pixel giữa các split. Các thông
tin so sánh `normal`–`other` bên dưới là kết quả audit của artifact cũ và là lý do đã loại
`normal`; chúng không mô tả dữ liệu dùng bởi checkpoint canonical hiện tại.

Lý do loại bỏ: 3.550 ảnh trong `color_filtered/other` trùng pixel với ảnh tương ứng trong
`vietnam_normal`. Khi hai nguồn được chia split độc lập, 1.315 cặp có một bản ở train và bản còn
lại ở validation/test, làm sai lệch đánh giá.

## Nguồn dữ liệu được giữ lại

| Nguồn | Loại biển |
|---|---|
| `dataset/color_filtered` | `blue`, `other`, `yellow` |
| `dataset/update_label/labled_quandoi` | `quandoi` |
| `dataset/update_label/label_ngoaigiao` | `ngoaigiao` |

Nguồn `vietnam_normal` bị loại hoàn toàn.

## Phân bố dữ liệu của checkpoint canonical

| Split | Blue | Ngoại giao | Other | Quân đội | Yellow | Tổng |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1 | 49 | 2.840 | 279 | 101 | 3.270 |
| Validation | 1 | 3 | 355 | 26 | 12 | 397 |
| Test | 1 | 5 | 355 | 38 | 12 | 411 |
| Tổng | 3 | 57 | 3.550 | 343 | 125 | 4.078 |

## Kết quả checkpoint được giữ

- Best epoch: 26.
- Validation exact match: 92,6952%.
- Test exact match với `refine_iters=2`: 91,9708%.
- Test character accuracy với `refine_iters=2`: 98,8684%.

## Bảo vệ trong notebook

Notebook khai báo `excluded_plate_types=("normal",)`. Hàm dựng dataframe:

1. bắt buộc cấu hình `dataset_sources` rõ ràng, không còn fallback ngầm sang `data_root/normal`;
2. bỏ qua nguồn được khai báo trực tiếp là `normal`;
3. lọc lại mọi row có `plate_type=normal` sau khi hợp nhất nguồn;
4. dừng với lỗi nếu không còn dữ liệu sau khi lọc.

## Kiểm tra split hiện tại và hướng cải tiến

Audit giải mã mọi ảnh về RGB rồi tính SHA-256 trên `shape + pixel bytes`. Kết quả có
`4.078/4.078` hash duy nhất và `0` nhóm ảnh trùng pixel chéo train/validation/test. Dữ liệu được
chia ở mức ảnh nên vẫn có thể có nhiều frame mang cùng chuỗi biển ở các split khác nhau. Đây không
phải bằng chứng leakage pixel của checkpoint hiện tại; group theo xe/track/source image là đề xuất
cho benchmark tổng quát hóa nghiêm ngặt hơn trong lần chuẩn bị dữ liệu tiếp theo.
